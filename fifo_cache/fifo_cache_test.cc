//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "plugin/fifo_cache/fifo_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "cache/lru_cache.h"
#include "file/file_util.h"
#include "port/port.h"
#include "rocksdb/cache.h"
#include "rocksdb/convenience.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/options_util.h"
#include "test_util/sync_point.h"
#include "test_util/testharness.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// Block cache keys are 16 bytes, the length the cache keeps as a
// reversible hash. Names up to 16 bytes, padded with zeros.
std::string K(const std::string& name) {
  EXPECT_LE(name.size(), size_t{16});
  std::string key = name;
  key.resize(16, '\0');
  return key;
}

std::string K(uint32_t i) {
  char buf[17];
  snprintf(buf, sizeof(buf), "key%013u", i);
  return std::string(buf, 16);
}

// Keys of several lengths, all distinct: by i % 3, "k" and i's digits (2
// to 11 bytes), K(i), or i's digits padded with 'x' to 17 + i % 100 bytes.
// Only the 16-byte ones skip the HeapKey.
std::string L(uint32_t i) {
  switch (i % 3) {
    case 0:
      return "k" + std::to_string(i);
    case 1:
      return K(i);
    default: {
      std::string key = std::to_string(i);
      key.resize(17 + i % 100, 'x');
      return key;
    }
  }
}

// Under kFullChargeCacheMetadata a key that isn't 16 bytes is charged for
// its copy: its length and a header of a helper pointer, a charge and a
// length.
constexpr size_t kHeapKeyHeader = sizeof(void*) + 2 * sizeof(size_t);

// Under kFullChargeCacheMetadata each slot the table has grown to is
// charged 64 bytes for itself, 4 for its probation word and 4 for its ghost
// tag, and each cell of the probation queue's rings 16.
constexpr size_t kSlotCharge = 64 + 4 + 4;
constexpr size_t kQueueCellCharge = 16;

// A charge unit for tests that count capacity in entries: 100 units of
// capacity hold 100 entries of one unit, far below the table's 1229 at its
// smallest, so only capacity evicts them.
constexpr size_t kU = 512;

// Values that count their deletions, to check that each value is deleted
// exactly once, and only when the cache lets go of it.
std::atomic<int64_t>& LiveValues() {
  static std::atomic<int64_t> count{0};
  return count;
}
constexpr uint32_t kLiveMagic = 0x1a7e1a7e;
constexpr uint32_t kDeadMagic = 0xdeaddead;
struct CountedValue {
  explicit CountedValue(uint32_t k) : magic(kLiveMagic), key(k) {
    LiveValues().fetch_add(1);
  }
  std::atomic<uint32_t> magic;
  const uint32_t key;
};
void DeleteCountedValue(Cache::ObjectPtr obj, MemoryAllocator* /*alloc*/) {
  auto* value = static_cast<CountedValue*>(obj);
  EXPECT_EQ(kLiveMagic, value->magic.exchange(kDeadMagic));
  LiveValues().fetch_sub(1);
  delete value;
}
const Cache::CacheItemHelper* CountedHelper() {
  static const Cache::CacheItemHelper helper{CacheEntryRole::kMisc,
                                             &DeleteCountedValue};
  return &helper;
}
void CheckCountedValue(Cache* cache, Cache::Handle* h, uint32_t k) {
  auto* value = static_cast<CountedValue*>(cache->Value(h));
  ASSERT_EQ(kLiveMagic, value->magic.load());
  ASSERT_EQ(k, value->key);
}

}  // namespace

class FIFOCacheTest : public testing::Test {
 public:
  // One shard and a fixed hash seed, so the table order is the same on
  // every host.
  std::shared_ptr<Cache> NewCache(
      size_t capacity, bool strict_capacity_limit = false,
      CacheMetadataChargePolicy policy = kDontChargeCacheMetadata,
      int num_shard_bits = 0, size_t max_capacity = 0) {
    FIFOCacheOptions opts(capacity, num_shard_bits, strict_capacity_limit,
                          /*memory_allocator*/ nullptr, kDefaultToAdaptiveMutex,
                          policy);
    opts.hash_seed = 12345;
    opts.max_capacity = max_capacity;
    std::shared_ptr<Cache> cache = opts.MakeSharedCache();
    EXPECT_NE(nullptr, cache);
    return cache;
  }

  static Status Insert(Cache* cache, const std::string& key, size_t charge = 1,
                       Cache::Handle** handle = nullptr,
                       Cache::Priority priority = Cache::Priority::LOW) {
    return cache->Insert(key, nullptr, &kNoopCacheItemHelper, charge, handle,
                         priority);
  }

  static FIFOCache* Fifo(Cache* cache) {
    auto* fifo = dynamic_cast<FIFOCache*>(cache);
    EXPECT_NE(nullptr, fifo);
    return fifo;
  }

  static bool Contains(Cache* cache, const std::string& key) {
    Cache::Handle* h = cache->Lookup(key);
    if (h == nullptr) {
      return false;
    }
    cache->Release(h);
    return true;
  }

  // The keys in the cache, without looking any up.
  static std::set<std::string> Keys(Cache* cache) {
    std::set<std::string> keys;
    cache->ApplyToAllEntries(
        [&](const Slice& key, Cache::ObjectPtr /*value*/, size_t /*charge*/,
            const Cache::CacheItemHelper* /*helper*/) {
          EXPECT_TRUE(keys.insert(key.ToString()).second);
        },
        {});
    return keys;
  }

  // Whether key is in the cache, without a Lookup (which would count as a
  // hit).
  static bool Holds(Cache* cache, const std::string& key) {
    return Keys(cache).count(key) != 0;
  }

  // Nine keys whose homes differ from each other's and from those of K(0) to
  // K(8) once the table has more than 64 slots (they differ in the low six
  // bits of the hash the home comes from), so none of their tags replaces
  // another's.
  static std::vector<std::string> NineKeysWithOwnHomes(Cache* cache) {
    std::set<uint64_t> taken;
    for (uint32_t i = 0; i < 9; ++i) {
      taken.insert(Shard::ComputeHash(K(i), cache->GetHashSeed())[1] & 63);
    }
    std::vector<std::string> nine;
    for (uint64_t low = 0; nine.size() < 9; ++low) {
      if (taken.count(low) == 0) {
        UniqueId64x2 unhashed;
        nine.push_back(Shard::ReverseHash(UniqueId64x2{0x5eed00 + low, low},
                                          &unhashed, cache->GetHashSeed())
                           .ToString());
      }
    }
    return nine;
  }

  // In a cache of 100 * kU, leaves the nine keys in the ghost with an entry
  // "big" of 79 * kU in the resident queue, hit three times, and 21 entries
  // of kU in probation, so that reinserting the nine evicts K(0) to K(8) from
  // probation and leaves it at its share. Reinserting K(8) then goes to the
  // resident queue, at no charge to probation, so the resident clock pays.
  static void GhostNineBehindBig(Cache* cache, FIFOCache* fifo,
                                 const std::vector<std::string>& nine) {
    ASSERT_EQ(9, nine.size());
    ASSERT_OK(Insert(cache, K("big"), 79 * kU));
    HitNTimes(cache, K("big"), 2);
    for (const std::string& key : nine) {
      ASSERT_OK(Insert(cache, key, kU));
    }
    for (uint32_t i = 0; i < 12; ++i) {
      ASSERT_OK(Insert(cache, K(i), kU));
    }
    ASSERT_EQ(100 * kU, cache->GetUsage());
    // The first promotes big, and each evicts one of the nine.
    for (uint32_t i = 100; i < 109; ++i) {
      ASSERT_OK(Insert(cache, K(i), kU));
    }
    for (const std::string& key : nine) {
      ASSERT_FALSE(Holds(cache, key));
      ASSERT_TRUE(fifo->TEST_InGhost(key)) << key.c_str();
    }
    ASSERT_FALSE(fifo->TEST_InProbation(K("big")));
    HitNTimes(cache, K("big"), 3);
    ASSERT_EQ(21 * kU, fifo->TEST_GetProbationUsage());
  }

  // HeapKeys allocated and not yet freed, in every cache. Release builds
  // don't count them and give 0.
  static int64_t LiveHeapKeys() {
#ifndef NDEBUG
    return fifo_cache::TEST_LiveHeapKeys();
#else
    return 0;
#endif  // !NDEBUG
  }

  // That n more HeapKeys are allocated than were at base, from LiveHeapKeys.
  // Always true in Release builds.
  static testing::AssertionResult HeapKeysAbove(int64_t base, int64_t n) {
#ifndef NDEBUG
    const int64_t more = fifo_cache::TEST_LiveHeapKeys() - base;
    if (more != n) {
      return testing::AssertionFailure()
             << more << " more HeapKeys, expected " << n;
    }
#else
    (void)base;
    (void)n;
#endif  // !NDEBUG
    return testing::AssertionSuccess();
  }

  using Shard = fifo_cache::ClockCacheShard<fifo_cache::AutoHyperClockTable>;

  // A shard on its own, so a test can give a key any hash.
  struct LoneShard {
    explicit LoneShard(size_t capacity, bool strict = false)
        : shard(capacity, strict, kDontChargeCacheMetadata,
                /*allocator*/ nullptr, &callback, &seed,
                fifo_cache::AutoHyperClockTable::Opts(
                    HyperClockCacheOptions(/*capacity*/ 0))) {}
    const Cache::EvictionCallback callback;
    const uint32_t seed = 12345;
    Shard shard;
  };

  static void HitNTimes(Cache* cache, const std::string& key, int n) {
    for (int i = 0; i < n; ++i) {
      Cache::Handle* h = cache->Lookup(key);
      ASSERT_NE(nullptr, h) << i;
      EXPECT_FALSE(cache->Release(h));
    }
  }

  // Keys of the entries the cache evicts, in order. Set before inserting
  // anything; evicted must outlive the cache.
  static void RecordEvictions(Cache* cache, std::vector<std::string>* evicted) {
    cache->SetEvictionCallback(
        [evicted](const Slice& key, Cache::Handle* /*h*/, bool /*was_hit*/) {
          evicted->push_back(key.ToString());
          return false;
        });
  }
};

TEST_F(FIFOCacheTest, MakeSharedCacheBasic) {
  FIFOCacheOptions opts(/*capacity*/ 10, /*num_shard_bits*/ 0,
                        /*strict_capacity_limit*/ false, nullptr,
                        kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  std::shared_ptr<Cache> cache = opts.MakeSharedCache();
  ASSERT_NE(nullptr, cache);
  EXPECT_STREQ("FIFOCache", cache->Name());
  EXPECT_NE(nullptr, dynamic_cast<FIFOCache*>(cache.get()));
  EXPECT_EQ(10, cache->GetCapacity());
  EXPECT_FALSE(cache->HasStrictCapacityLimit());
  EXPECT_OK(cache->Insert(K("k1"), reinterpret_cast<Cache::ObjectPtr>(42),
                          &kNoopCacheItemHelper, /*charge*/ 1));
  Cache::Handle* h = cache->Lookup(K("k1"));
  ASSERT_NE(nullptr, h);
  EXPECT_EQ(42, reinterpret_cast<uintptr_t>(cache->Value(h)));
  EXPECT_EQ(1, cache->GetCharge(h));
  EXPECT_EQ(&kNoopCacheItemHelper, cache->GetCacheItemHelper(h));
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_EQ(1, cache->GetPinnedUsage());
  EXPECT_EQ(1, cache->GetOccupancyCount());
  cache->Release(h);
  EXPECT_EQ(0, cache->GetPinnedUsage());
  cache->Erase(K("k1"));
  EXPECT_EQ(nullptr, cache->Lookup(K("k1")));
  EXPECT_EQ(0, cache->GetUsage());
  EXPECT_EQ(0, cache->GetOccupancyCount());
}

TEST_F(FIFOCacheTest, MakeSharedCacheValidation) {
  FIFOCacheOptions bad(/*capacity*/ 10, /*num_shard_bits*/ 20,
                       /*strict_capacity_limit*/ false, nullptr,
                       kDefaultToAdaptiveMutex,
                       kDefaultCacheMetadataChargePolicy);
  EXPECT_EQ(nullptr, bad.MakeSharedCache());

  // num_shard_bits left negative picks RocksDB's default for the capacity,
  // from MakeSharedCache and from the option string alike.
  for (size_t capacity : {size_t{1} << 20, size_t{64} << 20}) {
    FIFOCacheOptions dflt(capacity, /*num_shard_bits*/ -1,
                          /*strict_capacity_limit*/ false, nullptr,
                          kDefaultToAdaptiveMutex,
                          kDefaultCacheMetadataChargePolicy);
    std::shared_ptr<Cache> cache = dflt.MakeSharedCache();
    ASSERT_NE(nullptr, cache);
    auto* fifo = static_cast<FIFOCache*>(cache.get());
    EXPECT_EQ(GetDefaultCacheShardBits(capacity), fifo->GetNumShardBits());
    std::shared_ptr<Cache> from_string;
    ASSERT_OK(Cache::CreateFromString(
        ConfigOptions(), "fifo_cache://capacity=" + std::to_string(capacity),
        &from_string));
    EXPECT_EQ(GetDefaultCacheShardBits(capacity),
              static_cast<FIFOCache*>(from_string.get())->GetNumShardBits());
    FIFOCache direct(dflt);
    EXPECT_EQ(GetDefaultCacheShardBits(capacity), direct.GetNumShardBits());
    EXPECT_OK(Insert(cache.get(), K("a")));
    EXPECT_TRUE(Contains(cache.get(), K("a")));
  }
}

// Capacities smaller than a shard's starting table, down to zero, still
// build a working cache, under either metadata policy and with or without
// the strict limit. A non-strict insert that doesn't fit goes over capacity
// and the next one evicts it, so usage stays within one entry of the larger
// of capacity and the charge for the table's slots (which the table charges
// as it grows, even past capacity).
TEST_F(FIFOCacheTest, TinyCapacitiesBuildWorkingCaches) {
  struct Case {
    size_t capacity;
    int num_shard_bits;
  };
  for (const Case& c : {Case{0, 0}, Case{1, 0}, Case{63, 0}, Case{200, 0},
                        Case{300, 0}, Case{1000, 2}}) {
    for (CacheMetadataChargePolicy policy :
         {kFullChargeCacheMetadata, kDontChargeCacheMetadata}) {
      for (bool strict : {false, true}) {
        SCOPED_TRACE(std::to_string(c.capacity) + " " +
                     std::to_string(c.num_shard_bits) + " " +
                     std::to_string(policy) + " " + std::to_string(strict));
        std::shared_ptr<Cache> cache =
            NewCache(c.capacity, strict, policy, c.num_shard_bits);
        ASSERT_NE(nullptr, cache);
        for (uint32_t i = 0; i < 10; ++i) {
          Status s = Insert(cache.get(), K(i));
          EXPECT_TRUE(s.ok() || (strict && s.IsMemoryLimit())) << s.ToString();
        }
        if (!strict && c.num_shard_bits == 0) {
          const size_t metadata =
              policy == kFullChargeCacheMetadata
                  ? Fifo(cache.get())->TEST_GetMetadataCharge()
                  : 0;
          EXPECT_GE(metadata, policy == kFullChargeCacheMetadata
                                  ? cache->GetTableAddressCount() * kSlotCharge
                                  : 0);
          EXPECT_LE(cache->GetUsage(), std::max(c.capacity, metadata) + 1);
          EXPECT_GE(cache->GetOccupancyCount(), 1);
        }
        cache->EraseUnRefEntries();
        EXPECT_EQ(0, cache->GetOccupancyCount());
      }
    }
  }
}

// The constructor has no defaulted parameters, so positional calls written in
// the order of ShardedCacheOptions (fifth: metadata charge policy) or
// HyperClockCacheOptions (second: estimated entry charge) do not compile.
static_assert(
    !std::is_constructible_v<FIFOCacheOptions, size_t, int, bool,
                             std::nullptr_t, CacheMetadataChargePolicy>);
static_assert(
    !std::is_constructible_v<FIFOCacheOptions, size_t, size_t, int, bool,
                             std::nullptr_t, CacheMetadataChargePolicy>);

TEST_F(FIFOCacheTest, SecondaryCacheWrapped) {
  FIFOCacheOptions opts(/*capacity*/ 10, /*num_shard_bits*/ 0,
                        /*strict_capacity_limit*/ false, nullptr,
                        kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  opts.secondary_cache =
      CompressedSecondaryCacheOptions(/*capacity*/ 1000, /*num_shard_bits*/ 0,
                                      /*strict_capacity_limit*/ false, 0.5)
          .MakeSharedSecondaryCache();
  std::shared_ptr<Cache> cache = opts.MakeSharedCache();
  ASSERT_NE(nullptr, cache);
  EXPECT_EQ(nullptr, dynamic_cast<FIFOCache*>(cache.get()));
  auto* wrapper = dynamic_cast<CacheWrapper*>(cache.get());
  ASSERT_NE(nullptr, wrapper);
  EXPECT_STREQ("FIFOCache", wrapper->GetTarget()->Name());
  EXPECT_OK(cache->Insert(K("a"), nullptr, &kNoopCacheItemHelper, 1));
  Cache::Handle* h = cache->Lookup(K("a"));
  EXPECT_NE(nullptr, h);
  if (h != nullptr) {
    cache->Release(h);
  }
}

TEST_F(FIFOCacheTest, CreateFromStringRoundTripsScalarOptions) {
  ConfigOptions config_options;
  FIFOCacheOptions a(/*capacity*/ 12345, /*num_shard_bits*/ 3,
                     /*strict_capacity_limit*/ true, nullptr,
                     !kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  a.hash_seed = 42;
  a.max_capacity = 54321;
  FIFOCacheOptions b(/*capacity*/ size_t{1} << 30, /*num_shard_bits*/ 0,
                     /*strict_capacity_limit*/ false, nullptr,
                     kDefaultToAdaptiveMutex, kFullChargeCacheMetadata);
  b.hash_seed = ShardedCacheOptions::kQuasiRandomHashSeed;
  for (const FIFOCacheOptions& opts : {a, b}) {
    std::string str;
    ASSERT_OK(OptionTypeInfo::SerializeType(
        config_options, FIFOCacheOptionsTypeInfo(), &opts, &str));
    EXPECT_EQ(std::string::npos, str.find("memory_allocator"));
    EXPECT_EQ(std::string::npos, str.find("secondary_cache"));

    FIFOCacheOptions parsed;
    ASSERT_OK(OptionTypeInfo::ParseStruct(config_options, "fifo_cache",
                                          &FIFOCacheOptionsTypeInfo(),
                                          "fifo_cache", str, &parsed));
    EXPECT_EQ(opts.capacity, parsed.capacity);
    EXPECT_EQ(opts.num_shard_bits, parsed.num_shard_bits);
    EXPECT_EQ(opts.strict_capacity_limit, parsed.strict_capacity_limit);
    EXPECT_EQ(opts.metadata_charge_policy, parsed.metadata_charge_policy);
    EXPECT_EQ(opts.hash_seed, parsed.hash_seed);
    EXPECT_EQ(opts.use_adaptive_mutex, parsed.use_adaptive_mutex);
    EXPECT_EQ(opts.max_capacity, parsed.max_capacity);

    std::shared_ptr<Cache> cache;
    ASSERT_OK(
        Cache::CreateFromString(config_options, "fifo_cache://" + str, &cache));
    auto* fifo = dynamic_cast<FIFOCache*>(cache.get());
    ASSERT_NE(nullptr, fifo);
    EXPECT_EQ(opts.capacity, fifo->GetCapacity());
    EXPECT_EQ(opts.num_shard_bits, fifo->GetNumShardBits());
    EXPECT_EQ(opts.strict_capacity_limit, fifo->HasStrictCapacityLimit());
    if (opts.hash_seed >= 0) {
      EXPECT_EQ(static_cast<uint32_t>(opts.hash_seed), fifo->GetHashSeed());
    }
    ASSERT_OK(Insert(cache.get(), K("k")));
    EXPECT_EQ(opts.metadata_charge_policy == kDontChargeCacheMetadata,
              cache->GetUsage() == 1);
  }
}

// Spaces inside the braces are trimmed as in the LRU form, so a fifo_cache://
// value written that way builds a FIFOCache instead of falling through to the
// object registry, which returns OK and leaves the caller's cache unchanged.
// A plugin is reached only through the object registry, which matches the
// value from its start: in an option string the URI is the block_cache id,
// which replaces the default block cache, and takes a single option.
TEST_F(FIFOCacheTest, CreateFromStringTrimsSpacesInsideBraces) {
  ConfigOptions config_options;
  for (const char* value :
       {"{ id=fifo_cache://capacity=1M}", "{id=fifo_cache://capacity=1M }",
        "{  id = fifo_cache://capacity=1M }", " {id=fifo_cache://capacity=1M} ",
        "id=fifo_cache://capacity=1M"}) {
    BlockBasedTableOptions table_opts;
    ASSERT_OK(GetBlockBasedTableOptionsFromString(
        config_options, BlockBasedTableOptions(),
        std::string("block_cache=") + value, &table_opts))
        << value;
    const std::shared_ptr<Cache>& cache = table_opts.block_cache;
    ASSERT_NE(nullptr, cache) << value;
    EXPECT_STREQ("FIFOCache", cache->Name()) << value;
    EXPECT_EQ(size_t{1} << 20, cache->GetCapacity()) << value;
  }
  std::shared_ptr<Cache> cache;
  ASSERT_OK(Cache::CreateFromString(
      config_options, "fifo_cache://capacity=1M; num_shard_bits=2", &cache));
  ASSERT_NE(nullptr, cache);
  EXPECT_STREQ("FIFOCache", cache->Name());
  EXPECT_EQ(2, static_cast<FIFOCache*>(cache.get())->GetNumShardBits());

  Options options;
  ASSERT_OK(GetOptionsFromString(
      config_options, Options(),
      "block_based_table_factory={block_cache={id=fifo_cache://capacity=1M}}",
      &options));
  cache =
      options.table_factory->GetOptions<BlockBasedTableOptions>()->block_cache;
  ASSERT_NE(nullptr, cache);
  EXPECT_STREQ("FIFOCache", cache->Name());
  EXPECT_EQ(size_t{1} << 20, cache->GetCapacity());

  // More options than the id holds fail instead of being dropped.
  BlockBasedTableOptions table_opts;
  EXPECT_TRUE(GetBlockBasedTableOptionsFromString(
                  config_options, BlockBasedTableOptions(),
                  "block_cache={id=fifo_cache://capacity=1M;num_shard_bits=2}",
                  &table_opts)
                  .IsInvalidArgument());
}

// Invalid configuration fails with InvalidArgument naming the field, and
// leaves the output untouched. (The output starts empty: given an existing
// cache, RocksDB parses a value with '=' as options for that cache's own
// type, before the object registry is consulted.)
TEST_F(FIFOCacheTest, CreateFromStringRejectsInvalidOptions) {
  ConfigOptions config_options;
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"fifo_cache://", "capacity"},
      {"fifo_cache://num_shard_bits=2", "capacity"},
      {"fifo_cache://capacity=0", "capacity"},
      {"fifo_cache://capacity=1M;num_shard_bits=20", "num_shard_bits"},
      {"fifo_cache://capacity=1M;num_shard_bits=25", "num_shard_bits"},
      {"fifo_cache://capacity=abc", "capacity"},
      {"fifo_cache://capacity=1M;num_shard_bits=x", "num_shard_bits"},
      {"fifo_cache://capacity=1M;strict_capacity_limit=maybe",
       "strict_capacity_limit"},
      {"fifo_cache://capacity=1M;metadata_charge_policy=kSometimes",
       "metadata_charge_policy"},
      {"fifo_cache://capacity=1M;hash_seed=seed", "hash_seed"},
      {"fifo_cache://capacity=1M;use_adaptive_mutex=2", "use_adaptive_mutex"},
      {"fifo_cache://capacity=1M;max_capacity=abc", "max_capacity"},
      {"fifo_cache://capacity=1M;high_pri_pool_ratio=0.5",
       "high_pri_pool_ratio"},
      {"fifo_cache://capacity=1M;estimated_entry_charge=4096",
       "estimated_entry_charge"},
      {"fifo_cache://capacity=1M;probation_ratio=0.2", "probation_ratio"},
      {"fifo_cache://capacity=1M;memory_allocator=jemalloc",
       "memory_allocator"},
      {"fifo_cache://capacity=1M;"
       "secondary_cache=compressed_secondary_cache://capacity=1M",
       "secondary_cache"},
  };
  for (const auto& [value, field] : cases) {
    std::shared_ptr<Cache> cache;
    Status s = Cache::CreateFromString(config_options, value, &cache);
    EXPECT_TRUE(s.IsInvalidArgument()) << value << " -> " << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find(field))
        << value << " -> " << s.ToString();
    EXPECT_EQ(nullptr, cache) << value;
  }
}

// Strings that worked before never reach the fifo_cache:// branch: each
// builds the same LRUCache as the direct factory, and other URIs still go
// to the object registry.
TEST_F(FIFOCacheTest, CreateFromStringLeavesExistingStringsUnchanged) {
  ConfigOptions config_options;
  LRUCacheOptions full(/*capacity*/ 1 << 20, /*num_shard_bits*/ 4,
                       /*strict_capacity_limit*/ true,
                       /*high_pri_pool_ratio*/ 0.25);
  full.low_pri_pool_ratio = 0.125;
  const std::string full_str =
      "capacity=1M;num_shard_bits=4;strict_capacity_limit=true;"
      "high_pri_pool_ratio=0.25;low_pri_pool_ratio=0.125";
  const std::vector<std::pair<std::string, std::shared_ptr<Cache>>> cases = {
      {"1M", NewLRUCache(1 << 20)},
      {"capacity=2M", NewLRUCache(2 << 20)},
      {full_str, full.MakeSharedCache()},
      {"{" + full_str + "}", full.MakeSharedCache()},
  };
  for (const auto& [value, expected] : cases) {
    std::shared_ptr<Cache> direct;
    ASSERT_OK(Cache::CreateFromString(config_options, value, &direct));
    BlockBasedTableOptions table_opts;
    ASSERT_OK(GetBlockBasedTableOptionsFromString(
        config_options, BlockBasedTableOptions(),
        "block_cache=" +
            (value.find(';') == std::string::npos || value[0] == '{'
                 ? value
                 : "{" + value + "}"),
        &table_opts));
    for (const std::shared_ptr<Cache>& actual :
         {direct, table_opts.block_cache}) {
      auto* lru = dynamic_cast<LRUCache*>(actual.get());
      auto* want = static_cast<LRUCache*>(expected.get());
      ASSERT_NE(nullptr, lru) << value;
      EXPECT_STREQ("LRUCache", lru->Name());
      EXPECT_EQ(want->GetCapacity(), lru->GetCapacity()) << value;
      EXPECT_EQ(want->GetNumShardBits(), lru->GetNumShardBits()) << value;
      EXPECT_EQ(want->HasStrictCapacityLimit(), lru->HasStrictCapacityLimit());
      EXPECT_EQ(want->GetHashSeed(), lru->GetHashSeed()) << value;
      EXPECT_EQ(want->GetHighPriPoolRatio(), lru->GetHighPriPoolRatio());
      EXPECT_EQ(want->GetShard(0).GetLowPriPoolRatio(),
                lru->GetShard(0).GetLowPriPoolRatio());
    }
  }

  for (const char* uri : {"foo://bar", "my_fifo_cache://capacity=1"}) {
    std::shared_ptr<Cache> sentinel = NewLRUCache(1);
    std::shared_ptr<Cache> cache = sentinel;
    Status s = Cache::CreateFromString(config_options, uri, &cache);
    EXPECT_FALSE(s.IsInvalidArgument()) << s.ToString();
    EXPECT_EQ(sentinel, cache);
  }
}

// An erased entry that is still pinned stays readable through its handle,
// and charged, until its last release; Lookup no longer finds it.
TEST_F(FIFOCacheTest, ErasePinnedDetaches) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  ASSERT_OK(cache->Insert(K("a"), reinterpret_cast<Cache::ObjectPtr>(7),
                          &kNoopCacheItemHelper, /*charge*/ 3));
  Cache::Handle* h = cache->Lookup(K("a"));
  ASSERT_NE(nullptr, h);
  cache->Erase(K("a"));
  EXPECT_EQ(nullptr, cache->Lookup(K("a")));
  EXPECT_EQ(7, reinterpret_cast<uintptr_t>(cache->Value(h)));
  EXPECT_EQ(3, cache->GetUsage());
  EXPECT_EQ(3, cache->GetPinnedUsage());
  EXPECT_TRUE(cache->Release(h));
  EXPECT_EQ(0, cache->GetUsage());
  EXPECT_EQ(0, cache->GetOccupancyCount());
  // Erasing what isn't there changes nothing.
  ASSERT_OK(Insert(cache.get(), K("b")));
  cache->Erase(K("missing"));
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_TRUE(Contains(cache.get(), K("b")));
}

TEST_F(FIFOCacheTest, ReleaseEraseIfLastRef) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  ASSERT_OK(Insert(cache.get(), K("a"), /*charge*/ 2));
  Cache::Handle* h1 = cache->Lookup(K("a"));
  Cache::Handle* h2 = cache->Lookup(K("a"));
  ASSERT_NE(nullptr, h1);
  ASSERT_NE(nullptr, h2);
  // Not the last reference: nothing is erased.
  EXPECT_FALSE(cache->Release(h1, /*erase_if_last_ref*/ true));
  EXPECT_TRUE(Contains(cache.get(), K("a")));
  EXPECT_TRUE(cache->Release(h2, /*erase_if_last_ref*/ true));
  EXPECT_FALSE(Contains(cache.get(), K("a")));
  EXPECT_EQ(0, cache->GetUsage());
}

// Usage and pinned usage are the entries' charges, plus the table's under
// kFullChargeCacheMetadata. Pinned usage leaves out the slots of pinned
// entries under either policy, where HyperClockCache counts them.
TEST_F(FIFOCacheTest, UsageAndPinnedUsageAreExact) {
  for (CacheMetadataChargePolicy policy :
       {kDontChargeCacheMetadata, kFullChargeCacheMetadata}) {
    SCOPED_TRACE(policy);
    std::shared_ptr<Cache> cache =
        NewCache(/*capacity*/ 1 << 20, /*strict*/ false, policy);
    const size_t base = cache->GetUsage();
    ASSERT_OK(Insert(cache.get(), K("a"), /*charge*/ 5));
    Cache::Handle* hb = nullptr;
    ASSERT_OK(Insert(cache.get(), K("b"), /*charge*/ 7, &hb));
    ASSERT_NE(nullptr, hb);
    EXPECT_EQ(base + 12, cache->GetUsage());
    EXPECT_EQ(7, cache->GetPinnedUsage());
    Cache::Handle* ha = cache->Lookup(K("a"));
    ASSERT_NE(nullptr, ha);
    EXPECT_TRUE(cache->Ref(ha));
    EXPECT_EQ(12, cache->GetPinnedUsage());
    EXPECT_FALSE(cache->Release(ha));
    EXPECT_EQ(12, cache->GetPinnedUsage());
    EXPECT_FALSE(cache->Release(ha));
    EXPECT_EQ(7, cache->GetPinnedUsage());
    cache->Erase(K("b"));
    EXPECT_EQ(base + 12, cache->GetUsage());
    EXPECT_TRUE(cache->Release(hb));
    EXPECT_EQ(base + 5, cache->GetUsage());
    EXPECT_EQ(0, cache->GetPinnedUsage());
  }
}

// Under kFullChargeCacheMetadata usage counts every slot the table has grown
// to, with its probation word and ghost tag, and the probation queue's rings,
// besides the entries' charges; under kDontChargeCacheMetadata only the
// charges.
TEST_F(FIFOCacheTest, MetadataChargeCountsTableSlots) {
  for (CacheMetadataChargePolicy policy :
       {kFullChargeCacheMetadata, kDontChargeCacheMetadata}) {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 1 << 20,
                                            /*strict*/ false, policy);
    const size_t initial_slots = cache->GetTableAddressCount();
    size_t charges = 0;
    // Fewer entries than the table can hold for this capacity, so none is
    // evicted.
    for (uint32_t i = 0; i < 1000; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 1 + i % 7));
      charges += 1 + i % 7;
      const size_t metadata = policy == kFullChargeCacheMetadata
                                  ? Fifo(cache.get())->TEST_GetMetadataCharge()
                                  : 0;
      ASSERT_EQ(charges + metadata, cache->GetUsage()) << i;
    }
    EXPECT_GT(cache->GetTableAddressCount(), initial_slots);
    EXPECT_EQ(1000, cache->GetOccupancyCount());
    // With nothing evicted all 1000 entries are still in probation, so the
    // queue's rings doubled from 64 cells (the first table's occupancy limit
    // of 39, rounded up to a power of two) until one held the 1000th push.
    EXPECT_EQ(cache->GetTableAddressCount() * kSlotCharge +
                  (64 + 128 + 256 + 512 + 1024) * kQueueCellCharge,
              Fifo(cache.get())->TEST_GetMetadataCharge());
  }
}

// Pinned entries are never evicted, however many unpinned inserts follow,
// and the unpinned ones are evicted to stay within capacity.
TEST_F(FIFOCacheTest, PinnedEntriesAreNeverEvicted) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 20);
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 0; i < 10; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(cache->Insert(K(i), new CountedValue(i), CountedHelper(),
                            /*charge*/ 1, &h));
    pinned.push_back(h);
  }
  for (uint32_t i = 100; i < 1100; ++i) {
    ASSERT_OK(cache->Insert(K(i), new CountedValue(i), CountedHelper(),
                            /*charge*/ 1));
    ASSERT_LE(cache->GetUsage(), cache->GetCapacity()) << i;
  }
  for (uint32_t i = 0; i < 10; ++i) {
    CheckCountedValue(cache.get(), pinned[i], i);
    EXPECT_TRUE(Contains(cache.get(), K(i)));
  }
  EXPECT_EQ(10, cache->GetPinnedUsage());
  for (Cache::Handle* h : pinned) {
    cache->Release(h);
  }
  cache.reset();
  EXPECT_EQ(0, LiveValues().load());
}

// With every entry pinned, a non-strict insert goes over capacity instead
// of failing, with or without a handle, as in HyperClockCache. The next
// insert evicts what it can (here the unpinned c), and once the pins are
// released later inserts evict back under capacity.
TEST_F(FIFOCacheTest, NonStrictInsertOverFullyPinnedShardExceedsCapacity) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
  Cache::Handle* pa = nullptr;
  Cache::Handle* pb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("a"), 1, &pa));
  ASSERT_OK(Insert(cache.get(), K("b"), 1, &pb));
  ASSERT_OK(Insert(cache.get(), K("c")));
  EXPECT_EQ(3, cache->GetUsage());
  EXPECT_EQ((std::set<std::string>{K("a"), K("b"), K("c")}), Keys(cache.get()));
  Cache::Handle* pd = nullptr;
  ASSERT_OK(Insert(cache.get(), K("d"), 1, &pd));
  ASSERT_NE(nullptr, pd);
  EXPECT_EQ((std::set<std::string>{K("a"), K("b"), K("d")}), Keys(cache.get()));
  EXPECT_EQ(3, cache->GetUsage());
  EXPECT_EQ(3, cache->GetPinnedUsage());
  cache->Release(pa);
  cache->Release(pb);
  cache->Release(pd);
  for (uint32_t i = 0; i < 20; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  EXPECT_LE(cache->GetUsage(), 2);
}

// Under strict_capacity_limit an insert that asks for a handle fails when
// nothing can be evicted, and succeeds once something can.
TEST_F(FIFOCacheTest, StrictInsertWithHandleFailsWhenAllPinned) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2, /*strict*/ true);
  Cache::Handle* pa = nullptr;
  Cache::Handle* pb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("a"), 1, &pa));
  ASSERT_OK(Insert(cache.get(), K("b"), 1, &pb));
  Cache::Handle* pc = nullptr;
  Status s = Insert(cache.get(), K("c"), 1, &pc);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  EXPECT_EQ(nullptr, pc);
  EXPECT_FALSE(Contains(cache.get(), K("c")));
  EXPECT_EQ(2, cache->GetUsage());
  // An entry larger than the capacity never fits.
  cache->Release(pa);
  s = Insert(cache.get(), K("big"), 3, &pc);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  ASSERT_OK(Insert(cache.get(), K("c"), 1, &pc));
  ASSERT_NE(nullptr, pc);
  EXPECT_FALSE(Contains(cache.get(), K("a")));
  EXPECT_EQ(2, cache->GetUsage());
  cache->Release(pb);
  cache->Release(pc);
}

// Under strict_capacity_limit an insert without a handle that can't make
// room returns OK and is dropped, as if inserted and evicted at once,
// where HyperClockCache returns MemoryLimit: one that would fit beside
// some of the pins, and one larger than capacity. With a handle it fails.
// Once there is something to evict, it evicts.
TEST_F(FIFOCacheTest, StrictInsertWithoutHandleDropped) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10, /*strict*/ true);
  Cache::Handle* pa = nullptr;
  Cache::Handle* pb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("a"), 6, &pa));
  ASSERT_OK(Insert(cache.get(), K("b"), 4, &pb));
  ASSERT_OK(cache->Insert(K("c"), new CountedValue(0), CountedHelper(), 5));
  EXPECT_FALSE(Holds(cache.get(), K("c")));
  EXPECT_EQ(0, LiveValues().load());
  ASSERT_OK(Insert(cache.get(), K("big"), 11));
  EXPECT_FALSE(Holds(cache.get(), K("big")));
  Cache::Handle* pc = nullptr;
  Status s = Insert(cache.get(), K("c"), 5, &pc);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  EXPECT_EQ(nullptr, pc);
  EXPECT_EQ(10, cache->GetUsage());
  EXPECT_EQ(2, cache->GetOccupancyCount());
  cache->Release(pa);
  cache->Release(pb);
  ASSERT_OK(Insert(cache.get(), K("d"), 5));
  EXPECT_TRUE(Holds(cache.get(), K("d")));
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_TRUE(Holds(cache.get(), K("b")));
  EXPECT_EQ(9, cache->GetUsage());
}

// Under strict_capacity_limit with the table charged, the table grows only
// while the growth fits beside the entry. At the occupancy limit (39
// entries in the first 64 slots) an insert needs two slots, 65 and 66, to
// raise the limit to 40. With room for the entry alone it evicts an entry
// for its slot instead; with room for the two slots but not the entry too,
// it takes slot 65 (which fits beside the entry) and evicts; and with room
// for both it grows the table, landing usage on capacity. HyperClockCache
// charges its growth past capacity.
TEST_F(FIFOCacheTest, StrictInsertGrowsTableOnlyWhenGrowthFits) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ true, kFullChargeCacheMetadata);
  for (uint32_t i = 0; i < 39; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_EQ(64, cache->GetTableAddressCount());
  const size_t usage = cache->GetUsage();
  cache->SetCapacity(usage + 1);
  ASSERT_OK(Insert(cache.get(), K(100)));
  EXPECT_EQ(64, cache->GetTableAddressCount());
  EXPECT_EQ(39, cache->GetOccupancyCount());
  EXPECT_EQ(usage, cache->GetUsage());
  EXPECT_FALSE(Holds(cache.get(), K(0)));
  EXPECT_TRUE(Holds(cache.get(), K(100)));
  cache->SetCapacity(usage + 2 * kSlotCharge);
  ASSERT_OK(Insert(cache.get(), K(101)));
  EXPECT_EQ(65, cache->GetTableAddressCount());
  EXPECT_EQ(39, cache->GetOccupancyCount());
  EXPECT_EQ(usage + kSlotCharge, cache->GetUsage());
  EXPECT_FALSE(Holds(cache.get(), K(1)));
  EXPECT_TRUE(Holds(cache.get(), K(101)));
  cache->SetCapacity(usage + 1 + 2 * kSlotCharge);
  ASSERT_OK(Insert(cache.get(), K(102)));
  EXPECT_EQ(66, cache->GetTableAddressCount());
  EXPECT_EQ(40, cache->GetOccupancyCount());
  EXPECT_EQ(usage + 1 + 2 * kSlotCharge, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K(2)));
}

// Under strict_capacity_limit, with usage already over capacity (pins kept
// through a cut), an insert neither grows the table nor adds a ring: the
// metadata charge is turned down, and with nothing to evict the insert
// fails.
TEST_F(FIFOCacheTest, StrictMetadataNotChargedOverCapacity) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ true, kFullChargeCacheMetadata);
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 0; i < 64; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
    pins.push_back(h);
  }
  const size_t usage = cache->GetUsage();
  const size_t metadata = Fifo(cache.get())->TEST_GetMetadataCharge();
  cache->SetCapacity(usage - 1);
  ASSERT_EQ(usage, cache->GetUsage());
  Cache::Handle* h = nullptr;
  Status s = Insert(cache.get(), K(64), 1, &h);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  EXPECT_EQ(metadata, Fifo(cache.get())->TEST_GetMetadataCharge());
  EXPECT_EQ(usage, cache->GetUsage());
  for (Cache::Handle* p : pins) {
    cache->Release(p);
  }
}

// Under strict_capacity_limit with the table charged, the probation queue
// adds a ring only when it fits; until then an entry that finds the ring
// full goes to the resident queue. Here 64 entries fill the first ring of
// 64 cells, and the next ring has 128 (2 KiB).
TEST_F(FIFOCacheTest, StrictProbationRingOnlyWhenItFits) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ true, kFullChargeCacheMetadata);
  FIFOCache* fifo = Fifo(cache.get());
  for (uint32_t i = 0; i < 64; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  auto ring_cells = [&] {
    return (fifo->TEST_GetMetadataCharge() -
            cache->GetTableAddressCount() * kSlotCharge) /
           kQueueCellCharge;
  };
  ASSERT_EQ(64, ring_cells());
  // Room for the entry and for growing the table, not for the ring.
  cache->SetCapacity(cache->GetUsage() + 1 + 4 * kSlotCharge);
  ASSERT_OK(Insert(cache.get(), K(64)));
  EXPECT_EQ(64, ring_cells());
  EXPECT_TRUE(Holds(cache.get(), K(64)));
  EXPECT_FALSE(fifo->TEST_InProbation(K(64)));
  EXPECT_LE(cache->GetUsage(), cache->GetCapacity());
  cache->SetCapacity(cache->GetUsage() + 1 + 4 * kSlotCharge +
                     128 * kQueueCellCharge);
  ASSERT_OK(Insert(cache.get(), K(65)));
  EXPECT_EQ(64 + 128, ring_cells());
  EXPECT_TRUE(fifo->TEST_InProbation(K(65)));
  EXPECT_LE(cache->GetUsage(), cache->GetCapacity());
}

// Under strict_capacity_limit with the table charged, capacities from just
// above the first table's charge (4 slots and a 4-cell ring) up keep
// admitting new entries, with or without a handle, evicting for a slot
// where the growth doesn't fit, and usage never passes capacity.
TEST_F(FIFOCacheTest, StrictInsertEvictsWhereGrowthDoesNotFit) {
  const size_t base = 4 * kSlotCharge + 4 * kQueueCellCharge;
  for (size_t capacity = base + 1; capacity <= base + 1200; capacity += 4) {
    SCOPED_TRACE(capacity);
    std::shared_ptr<Cache> cache =
        NewCache(capacity, /*strict*/ true, kFullChargeCacheMetadata);
    ASSERT_EQ(base, cache->GetUsage());
    for (uint32_t i = 0; i < 64; ++i) {
      if (i % 2 == 0) {
        Cache::Handle* h = nullptr;
        ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
        EXPECT_FALSE(cache->Release(h));
      } else {
        ASSERT_OK(Insert(cache.get(), K(i)));
        EXPECT_TRUE(Holds(cache.get(), K(i)));
      }
      ASSERT_LE(cache->GetUsage(), capacity);
    }
  }
}

// SetStrictCapacityLimit reaches every shard and takes effect on the next
// insert, either way.
TEST_F(FIFOCacheTest, SetStrictCapacityLimitTogglesInserts) {
  std::shared_ptr<Cache> cache =
      NewCache(/*capacity*/ 2, /*strict*/ false, kDontChargeCacheMetadata,
               /*num_shard_bits*/ 1);
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 0; pins.size() < 4; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
    pins.push_back(h);
  }
  EXPECT_EQ(4, cache->GetUsage());
  cache->SetStrictCapacityLimit(true);
  for (uint32_t i = 100; i < 120; ++i) {
    Cache::Handle* h = nullptr;
    Status s = Insert(cache.get(), K(i), 1, &h);
    EXPECT_TRUE(s.IsMemoryLimit()) << i << " " << s.ToString();
    EXPECT_EQ(nullptr, h);
  }
  cache->SetStrictCapacityLimit(false);
  Cache::Handle* h = nullptr;
  ASSERT_OK(Insert(cache.get(), K(200), 1, &h));
  ASSERT_NE(nullptr, h);
  pins.push_back(h);
  EXPECT_EQ(5, cache->GetUsage());
  for (Cache::Handle* p : pins) {
    cache->Release(p);
  }
}

// Charges near SIZE_MAX don't wrap usage under either limit: such an
// entry is refused with a handle and dropped without one, and a
// standalone one is refused, or left uncharged where that's allowed.
TEST_F(FIFOCacheTest, HugeChargesDoNotWrapAccounting) {
  constexpr size_t kHuge = SIZE_MAX - 10;
  for (bool strict : {true, false}) {
    SCOPED_TRACE(strict);
    std::shared_ptr<Cache> cache =
        NewCache(1 << 20, strict, kFullChargeCacheMetadata);
    Cache::Handle* pinned = nullptr;
    ASSERT_OK(Insert(cache.get(), K("p"), 20, &pinned));
    const size_t usage = cache->GetUsage();
    Cache::Handle* h = nullptr;
    Status s = Insert(cache.get(), K("huge"), kHuge, &h);
    EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
    EXPECT_EQ(nullptr, h);
    EXPECT_EQ(usage, cache->GetUsage());
    ASSERT_OK(Insert(cache.get(), K("huge2"), kHuge));
    EXPECT_FALSE(Holds(cache.get(), K("huge2")));
    EXPECT_EQ(usage, cache->GetUsage());
    // Under half of size_t, but not with the usage already there.
    s = Insert(cache.get(), K("half"), SIZE_MAX / 2 - usage + 1, &h);
    EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
    EXPECT_EQ(nullptr, h);
    EXPECT_EQ(usage, cache->GetUsage());
    // A 1-byte key's charge saturates the entry's.
    h = cache->CreateStandalone("s", nullptr, &kNoopCacheItemHelper, kHuge,
                                /*allow_uncharged*/ true);
    ASSERT_NE(nullptr, h);
    EXPECT_EQ(0, cache->GetCharge(h));
    EXPECT_EQ(usage, cache->GetUsage());
    EXPECT_TRUE(cache->Release(h));
    EXPECT_EQ(usage, cache->GetUsage());
    EXPECT_EQ(nullptr,
              cache->CreateStandalone(K("t"), nullptr, &kNoopCacheItemHelper,
                                      kHuge, /*allow_uncharged*/ false));
    EXPECT_EQ(usage, cache->GetUsage());
    EXPECT_EQ(1, cache->GetOccupancyCount());
    EXPECT_FALSE(cache->Release(pinned));
  }
}

TEST_F(FIFOCacheTest, CreateStandalone) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 5, /*strict*/ true);
  Cache::Handle* h = cache->CreateStandalone(K("s"), nullptr,
                                             &kNoopCacheItemHelper,
                                             /*charge*/ 4,
                                             /*allow_uncharged*/ false);
  ASSERT_NE(nullptr, h);
  EXPECT_FALSE(Contains(cache.get(), K("s")));
  EXPECT_EQ(4, cache->GetUsage());
  EXPECT_EQ(4, cache->GetPinnedUsage());
  EXPECT_EQ(0, cache->GetOccupancyCount());
  // No room left: refused, or created without a charge when allowed.
  EXPECT_EQ(nullptr,
            cache->CreateStandalone(K("t"), nullptr, &kNoopCacheItemHelper,
                                    /*charge*/ 3, /*allow_uncharged*/ false));
  Cache::Handle* u =
      cache->CreateStandalone(K("u"), nullptr, &kNoopCacheItemHelper,
                              /*charge*/ 3, /*allow_uncharged*/ true);
  ASSERT_NE(nullptr, u);
  EXPECT_EQ(0, cache->GetCharge(u));
  EXPECT_EQ(4, cache->GetUsage());
  EXPECT_TRUE(cache->Release(u));
  EXPECT_TRUE(cache->Release(h));
  EXPECT_EQ(0, cache->GetUsage());
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// Cache::Priority is ignored: entries inserted with HIGH priority are
// evicted alongside BOTTOM ones. (HyperClockCache starts BOTTOM entries with
// a lower count, so it evicts them first.)
TEST_F(FIFOCacheTest, PriorityIgnored) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 40);
  for (uint32_t i = 0; i < 20; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), 1, nullptr, Cache::Priority::HIGH));
    ASSERT_OK(
        Insert(cache.get(), K(100 + i), 1, nullptr, Cache::Priority::BOTTOM));
  }
  for (uint32_t i = 200; i < 220; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  int high_evicted = 0;
  int bottom_evicted = 0;
  const std::set<std::string> keys = Keys(cache.get());
  for (uint32_t i = 0; i < 20; ++i) {
    high_evicted += keys.count(K(i)) == 0;
    bottom_evicted += keys.count(K(100 + i)) == 0;
  }
  EXPECT_GT(high_evicted, 0);
  EXPECT_GT(bottom_evicted, 0);
}

TEST_F(FIFOCacheTest, LookupIgnoresSecondaryParameters) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  ASSERT_OK(Insert(cache.get(), K("a")));
  Cache::CreateContext context;
  // A secondary-compatible helper whose create callback must not run: the
  // cache has no secondary cache of its own.
  Cache::CacheItemHelper secondary_helper(
      CacheEntryRole::kMisc, nullptr,
      [](Cache::ObjectPtr) -> size_t { return 0; },
      [](Cache::ObjectPtr, size_t, size_t, char*) { return Status::OK(); },
      [](const Slice&, CompressionType, CacheTier, Cache::CreateContext*,
         MemoryAllocator*, Cache::ObjectPtr*, size_t*) {
        ADD_FAILURE() << "create callback called";
        return Status::NotSupported();
      },
      &kNoopCacheItemHelper);
  Cache::Handle* h = cache->Lookup(K("a"), &secondary_helper, &context,
                                   Cache::Priority::HIGH, nullptr);
  ASSERT_NE(nullptr, h);
  cache->Release(h);
  EXPECT_EQ(nullptr, cache->Lookup(K("missing"), &secondary_helper, &context,
                                   Cache::Priority::HIGH, nullptr));
}

// A 16-byte key is stored as a reversible hash and any other in a HeapKey;
// ApplyToAllEntries and the eviction callback give back the exact key of
// every length, the empty key too, under a nonzero seed and on several
// shards.
TEST_F(FIFOCacheTest, KeysComeBackExact) {
  FIFOCacheOptions opts(/*capacity*/ 1 << 20, /*num_shard_bits*/ 2,
                        /*strict_capacity_limit*/ false, nullptr,
                        kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  opts.hash_seed = 12345;
  std::shared_ptr<Cache> cache = opts.MakeSharedCache();
  ASSERT_NE(nullptr, cache);
  std::set<std::string> inserted = {std::string()};
  ASSERT_OK(Insert(cache.get(), std::string(), /*charge*/ 1000));
  for (uint32_t i = 0; i < 200; ++i) {
    inserted.insert(L(i));
    ASSERT_OK(Insert(cache.get(), L(i), /*charge*/ 1000));
  }
  EXPECT_EQ(inserted, Keys(cache.get()));
  for (const std::string& key : inserted) {
    EXPECT_TRUE(Contains(cache.get(), key));
  }

  // Each shard holds 256 KiB, about 260 of these: evict by inserting more.
  std::set<std::string> evicted;
  cache->SetEvictionCallback(
      [&](const Slice& key, Cache::Handle* /*h*/, bool /*was_hit*/) {
        EXPECT_TRUE(evicted.insert(key.ToString()).second);
        return false;
      });
  for (uint32_t i = 1000; i < 3000; ++i) {
    inserted.insert(L(i));
    ASSERT_OK(Insert(cache.get(), L(i), /*charge*/ 1000));
  }
  const std::set<std::string> remaining = Keys(cache.get());
  EXPECT_FALSE(evicted.empty());
  std::set<std::string> all = evicted;
  for (const std::string& key : remaining) {
    EXPECT_EQ(0, evicted.count(key));
    all.insert(key);
  }
  EXPECT_EQ(inserted, all);
}

// The eviction callback runs for evictions, not for Erase or
// EraseUnRefEntries, and a callback that returns true takes the value
// instead of the cache deleting it. The cache still frees a HeapKey.
TEST_F(FIFOCacheTest, EvictionCallbackTakesValues) {
  std::vector<CountedValue*> taken;
  const int64_t heap_keys = LiveHeapKeys();
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 5);
    Cache* raw = cache.get();
    cache->SetEvictionCallback(
        [&taken, raw](const Slice& key, Cache::Handle* h, bool /*was_hit*/) {
          auto* value = static_cast<CountedValue*>(raw->Value(h));
          EXPECT_EQ(L(value->key), key.ToString());
          taken.push_back(value);
          return true;
        });
    ASSERT_OK(cache->Insert(L(0), new CountedValue(0), CountedHelper(), 1));
    cache->Erase(L(0));
    EXPECT_TRUE(taken.empty());
    for (uint32_t i = 1; i <= 20; ++i) {
      ASSERT_OK(cache->Insert(L(i), new CountedValue(i), CountedHelper(), 1));
    }
    EXPECT_GE(taken.size(), 15);
    for (CountedValue* value : taken) {
      EXPECT_EQ(kLiveMagic, value->magic.load());
    }
    const size_t evicted = taken.size();
    cache->EraseUnRefEntries();
    EXPECT_EQ(evicted, taken.size());
  }
  // The cache deleted everything but the values the callback took, and
  // freed every key.
  EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
  EXPECT_EQ(static_cast<int64_t>(taken.size()), LiveValues().load());
  for (CountedValue* value : taken) {
    DeleteCountedValue(value, nullptr);
  }
  EXPECT_EQ(0, LiveValues().load());
}

TEST_F(FIFOCacheTest, EraseUnRefEntriesKeepsPinned) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 100);
  for (uint32_t i = 0; i < 5; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  Cache::Handle* pinned = cache->Lookup(K(0));
  ASSERT_NE(nullptr, pinned);
  cache->EraseUnRefEntries();
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_EQ(1, cache->GetOccupancyCount());
  EXPECT_FALSE(Contains(cache.get(), K(1)));
  cache->Release(pinned);
  EXPECT_TRUE(Contains(cache.get(), K(0)));
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// Keys are cached and found by their bytes whatever their length: the
// empty key, keys that differ only in length or in one byte, and a 16-byte
// key next to its 17-byte extension. Erasing one leaves the others.
TEST_F(FIFOCacheTest, KeysOfAnyLengthAreKeptApart) {
  const int64_t heap_keys = LiveHeapKeys();
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 1 << 20);
    const std::vector<std::string> keys = {
        std::string(), "a",    std::string("a\0", 2), "b",
        "ab",          K("a"), K("a") + '\0',         std::string(1000, 'a')};
    for (uint32_t i = 0; i < keys.size(); ++i) {
      ASSERT_OK(
          cache->Insert(keys[i], new CountedValue(i), CountedHelper(), 1));
    }
    EXPECT_TRUE(HeapKeysAbove(heap_keys, keys.size() - 1));
    EXPECT_EQ(std::set<std::string>(keys.begin(), keys.end()),
              Keys(cache.get()));
    for (uint32_t i = 0; i < keys.size(); ++i) {
      Cache::Handle* h = cache->Lookup(keys[i]);
      ASSERT_NE(nullptr, h) << i;
      CheckCountedValue(cache.get(), h, i);
      cache->Release(h);
    }
    for (const std::string& absent :
         {std::string("c"), std::string("a\1", 2), std::string(15, 'a'),
          std::string(999, 'a')}) {
      EXPECT_EQ(nullptr, cache->Lookup(absent));
    }
    for (uint32_t i = 0; i < keys.size(); ++i) {
      cache->Erase(keys[i]);
      for (uint32_t j = 0; j < keys.size(); ++j) {
        EXPECT_EQ(j > i, Contains(cache.get(), keys[j])) << i << " " << j;
      }
    }
    EXPECT_EQ(0, LiveValues().load());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
  }
}

// Keys with the same 128-bit hash, one of 16 bytes and twelve that aren't,
// share a home in a shard. Each is found by its own bytes, including those
// past the ten entries Lookup checks before its full search, each insert
// goes in although an entry with the hash is at the home, and Erase takes
// only the key it names.
TEST_F(FIFOCacheTest, KeysWithOneHashAreKeptApart) {
  const int64_t heap_keys = LiveHeapKeys();
  {
    LoneShard lone(/*capacity*/ 1 << 20);
    Shard& shard = lone.shard;
    const UniqueId64x2 hash = Shard::ComputeHash(K("c"), lone.seed);
    std::vector<std::string> keys = {K("c")};
    for (uint32_t i = 1; i <= 12; ++i) {
      keys.push_back("h" + std::to_string(i));
    }
    for (uint32_t i = 0; i < keys.size(); ++i) {
      ASSERT_OK(shard.Insert(keys[i], hash, new CountedValue(i),
                             CountedHelper(), /*charge*/ 1,
                             /*handle*/ nullptr, Cache::Priority::LOW));
    }
    EXPECT_EQ(keys.size(), shard.GetOccupancyCount());
    // The value's number, or -1 if the key isn't found.
    auto find = [&](const std::string& key) -> int64_t {
      Shard::HandleImpl* h = shard.Lookup(key, hash);
      if (h == nullptr) {
        return -1;
      }
      const int64_t found = static_cast<CountedValue*>(h->value)->key;
      shard.Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
      return found;
    };
    for (uint32_t i = 0; i < keys.size(); ++i) {
      EXPECT_EQ(i, find(keys[i])) << i;
    }
    EXPECT_EQ(-1, find("h13"));
    EXPECT_EQ(-1, find(std::string()));
    shard.Erase(keys[0], hash);
    shard.Erase(keys[5], hash);
    for (uint32_t i = 0; i < keys.size(); ++i) {
      EXPECT_EQ(i == 0 || i == 5 ? -1 : int64_t{i}, find(keys[i])) << i;
    }
    EXPECT_EQ(keys.size() - 2, shard.GetOccupancyCount());
    EXPECT_EQ(keys.size() - 2, LiveValues().load());
  }
  EXPECT_EQ(0, LiveValues().load());
  EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
}

// Under kFullChargeCacheMetadata a key that isn't 16 bytes adds its copy's
// size to usage and pinned usage, and under kDontChargeCacheMetadata
// nothing. GetCharge, ApplyToHandle and ApplyToAllEntries give the charge
// the entry was inserted with, and its helper comes back.
TEST_F(FIFOCacheTest, HeapKeyChargedAsMetadata) {
  for (CacheMetadataChargePolicy policy :
       {kDontChargeCacheMetadata, kFullChargeCacheMetadata}) {
    SCOPED_TRACE(policy);
    const size_t key_charge =
        policy == kFullChargeCacheMetadata ? kHeapKeyHeader + 3 : 0;
    std::shared_ptr<Cache> cache = NewCache(1 << 20, /*strict*/ false, policy);
    const size_t base = cache->GetUsage();
    ASSERT_OK(Insert(cache.get(), K("a"), /*charge*/ 10));
    EXPECT_EQ(base + 10, cache->GetUsage());
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), "abc", /*charge*/ 20, &h));
    EXPECT_EQ(base + 30 + key_charge, cache->GetUsage());
    EXPECT_EQ(20 + key_charge, cache->GetPinnedUsage());
    EXPECT_EQ(20, cache->GetCharge(h));
    EXPECT_EQ(&kNoopCacheItemHelper, cache->GetCacheItemHelper(h));
    int applied = 0;
    cache->ApplyToHandle(
        cache.get(), h,
        [&](const Slice& key, Cache::ObjectPtr /*value*/, size_t charge,
            const Cache::CacheItemHelper* helper) {
          ++applied;
          EXPECT_EQ("abc", key.ToString());
          EXPECT_EQ(20, charge);
          EXPECT_EQ(&kNoopCacheItemHelper, helper);
        });
    EXPECT_EQ(1, applied);
    std::map<std::string, size_t> charges;
    cache->ApplyToAllEntries(
        [&](const Slice& key, Cache::ObjectPtr /*value*/, size_t charge,
            const Cache::CacheItemHelper* helper) {
          EXPECT_EQ(&kNoopCacheItemHelper, helper);
          charges[key.ToString()] = charge;
        },
        {});
    EXPECT_EQ((std::map<std::string, size_t>{{K("a"), 10}, {"abc", 20}}),
              charges);
    EXPECT_FALSE(cache->Release(h));
    EXPECT_EQ(0, cache->GetPinnedUsage());
    cache->Erase("abc");
    EXPECT_EQ(base + 10, cache->GetUsage());
  }
}

// Each way an entry with a HeapKey leaves the cache frees its copy once
// and takes the copy's charge out of usage: Erase, unpinned or pinned and
// then released, a release that erases, an insert replacing it, unpinned
// or pinned and then released, a standalone entry's release,
// EraseUnRefEntries, eviction and the cache's destruction.
TEST_F(FIFOCacheTest, HeapKeyFreedOnEveryRemovalPath) {
  const int64_t heap_keys = LiveHeapKeys();
  {
    std::shared_ptr<Cache> cache =
        NewCache(64 * kU, /*strict*/ false, kFullChargeCacheMetadata);
    Cache* c = cache.get();
    FIFOCache* fifo = Fifo(c);
    const size_t base = fifo->TEST_GetMetadataCharge();
    ASSERT_EQ(base, c->GetUsage());
    // A one-byte key.
    const size_t charged = kU + kHeapKeyHeader + 1;

    ASSERT_OK(Insert(c, "a", kU));
    EXPECT_EQ(base + charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));
    c->Erase("a");
    EXPECT_EQ(base, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));

    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(c, "a", kU, &h));
    c->Erase("a");
    EXPECT_EQ(base + charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));
    EXPECT_TRUE(c->Release(h));
    EXPECT_EQ(base, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));

    ASSERT_OK(Insert(c, "a", kU, &h));
    EXPECT_TRUE(c->Release(h, /*erase_if_last_ref*/ true));
    EXPECT_EQ(base, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));

    ASSERT_OK(Insert(c, "a", kU));
    Status s = Insert(c, "a", kU);
    ASSERT_TRUE(s.IsOkOverwritten()) << s.ToString();
    EXPECT_EQ(base + charged, c->GetUsage());
    EXPECT_EQ(1, c->GetOccupancyCount());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));
    Cache::Handle* old = c->Lookup("a");
    ASSERT_NE(nullptr, old);
    s = Insert(c, "a", kU);
    ASSERT_TRUE(s.IsOkOverwritten()) << s.ToString();
    EXPECT_EQ(base + 2 * charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 2));
    EXPECT_TRUE(c->Release(old));
    EXPECT_EQ(base + charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));

    h = c->CreateStandalone("s", nullptr, &kNoopCacheItemHelper, kU,
                            /*allow_uncharged*/ false);
    ASSERT_NE(nullptr, h);
    EXPECT_EQ(base + 2 * charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 2));
    EXPECT_TRUE(c->Release(h));
    EXPECT_EQ(base + charged, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));

    c->EraseUnRefEntries();
    EXPECT_EQ(base, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));

    std::vector<std::string> evicted;
    RecordEvictions(c, &evicted);
    size_t inserted = 0;
    for (uint32_t i = 0; i < 100; ++i) {
      ASSERT_OK(Insert(c, "e" + std::to_string(i), kU));
      inserted += kU + kHeapKeyHeader + 1 + std::to_string(i).size();
    }
    ASSERT_FALSE(evicted.empty());
    size_t freed = 0;
    for (const std::string& key : evicted) {
      freed += kU + kHeapKeyHeader + key.size();
    }
    EXPECT_EQ(fifo->TEST_GetMetadataCharge() + inserted - freed, c->GetUsage());
    EXPECT_EQ(100 - evicted.size(), c->GetOccupancyCount());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 100 - evicted.size()));
  }
  EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
}

// An insert with a handle or CreateStandalone that strict_capacity_limit
// refuses frees the key's copy, charges nothing and leaves the value with
// the caller, and one without a handle is dropped, freeing both. An
// uncharged standalone entry charges nothing for its key either, and
// reports a charge of 0.
TEST_F(FIFOCacheTest, HeapKeyFreedWhenStrictInsertFails) {
  const int64_t heap_keys = LiveHeapKeys();
  {
    std::shared_ptr<Cache> cache =
        NewCache(1 << 20, /*strict*/ true, kFullChargeCacheMetadata);
    Cache* c = cache.get();
    Cache::Handle* pinned = nullptr;
    ASSERT_OK(Insert(c, K("p"), (1 << 20) - c->GetUsage(), &pinned));
    ASSERT_EQ(1 << 20, c->GetUsage());

    auto* value = new CountedValue(1);
    Cache::Handle* h = nullptr;
    Status s = c->Insert("a", value, CountedHelper(), 1, &h);
    EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
    EXPECT_EQ(nullptr, h);
    EXPECT_EQ(kLiveMagic, value->magic.load());
    DeleteCountedValue(value, nullptr);
    // Without a handle, dropped: the cache frees the value and the key.
    ASSERT_OK(c->Insert("a", new CountedValue(2), CountedHelper(), 1));
    EXPECT_EQ(0, LiveValues().load());
    EXPECT_EQ(nullptr,
              c->CreateStandalone("a", nullptr, &kNoopCacheItemHelper, 1,
                                  /*allow_uncharged*/ false));
    EXPECT_EQ(1 << 20, c->GetUsage());
    EXPECT_EQ(1, c->GetOccupancyCount());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));

    h = c->CreateStandalone("s", nullptr, &kNoopCacheItemHelper, 3,
                            /*allow_uncharged*/ true);
    ASSERT_NE(nullptr, h);
    EXPECT_EQ(0, c->GetCharge(h));
    EXPECT_EQ(1 << 20, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 1));
    EXPECT_TRUE(c->Release(h));
    EXPECT_EQ(1 << 20, c->GetUsage());
    EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
    EXPECT_FALSE(c->Release(pinned));
  }
}

// The key's charge is added to the entry's without wrapping, so an insert
// with a charge near SIZE_MAX is still too big for the cache.
TEST_F(FIFOCacheTest, HeapKeyChargeSaturates) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ true, kFullChargeCacheMetadata);
  const size_t usage = cache->GetUsage();
  Cache::Handle* h = nullptr;
  Status s = Insert(cache.get(), "abc", SIZE_MAX - 10, &h);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  EXPECT_EQ(nullptr, h);
  EXPECT_EQ(usage, cache->GetUsage());
  EXPECT_EQ(0, cache->GetOccupancyCount());
}

// A copy of the key that can't be allocated fails Insert with MemoryLimit
// and CreateStandalone with nullptr, caching and charging nothing: a length
// whose copy wouldn't fit in a size_t is refused before malloc, and malloc
// refuses one of half that. The key's bytes are never read, as the shard
// is given the hash.
TEST_F(FIFOCacheTest, KeyAllocationFailureFailsInsert) {
  LoneShard lone(/*capacity*/ 1 << 20);
  const char byte = 'k';
  std::vector<size_t> sizes = {SIZE_MAX - 1};
#if !defined(MUST_FREE_HEAP_ALLOCATIONS) && !defined(__SANITIZE_THREAD__)
  // Sanitizer allocators abort on such a request rather than return nullptr.
  sizes.push_back(SIZE_MAX / 2);
#endif
  for (size_t size : sizes) {
    SCOPED_TRACE(size);
    const Slice huge(&byte, size);
    const UniqueId64x2 hash{{1, 2}};
    Shard::HandleImpl* h = nullptr;
    Status s =
        lone.shard.Insert(huge, hash, /*value*/ nullptr, &kNoopCacheItemHelper,
                          /*charge*/ 1, &h, Cache::Priority::LOW);
    EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
    EXPECT_EQ(nullptr, h);
    EXPECT_EQ(nullptr,
              lone.shard.CreateStandalone(huge, hash, /*value*/ nullptr,
                                          &kNoopCacheItemHelper, /*charge*/ 1,
                                          /*allow_uncharged*/ true));
  }
  EXPECT_EQ(0, lone.shard.GetUsage());
  EXPECT_EQ(0, lone.shard.GetOccupancyCount());
}

// A key that isn't 16 bytes leaves a ghost when evicted from probation and
// is readmitted by it. Its placeholder, which kFullChargeCacheMetadata
// charges for the key's copy, still neither takes the ghost nor goes to
// the resident queue.
TEST_F(FIFOCacheTest, GhostKeepsKeysOfAnyLength) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ false, kFullChargeCacheMetadata);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), "a", kU));
  for (uint32_t i = 0; i < 9; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  cache->SetCapacity(cache->GetUsage());
  // Evicts a.
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_FALSE(Holds(cache.get(), "a"));
  ASSERT_TRUE(fifo->TEST_InGhost("a"));
  ASSERT_OK(Insert(cache.get(), "a", /*charge*/ 0));
  EXPECT_TRUE(fifo->TEST_InProbation("a"));
  EXPECT_TRUE(fifo->TEST_InGhost("a"));
  cache->Erase("a");
  ASSERT_OK(Insert(cache.get(), "a", kU));
  EXPECT_TRUE(Holds(cache.get(), "a"));
  EXPECT_FALSE(fifo->TEST_InProbation("a"));
  EXPECT_FALSE(fifo->TEST_InGhost("a"));
}

// ApplyToAllEntries visits the slots a batch at a time without holding
// anything between batches, so the table can grow part way through a scan.
// Entries never move when it does: every key present from start to end is
// visited exactly once, and no key twice.
TEST_F(FIFOCacheTest, ApplyToAllEntriesCoversEachKeyOnceAcrossGrowth) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 1 << 20);
  for (uint32_t i = 0; i < 15; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  const size_t length_before = cache->GetTableAddressCount();
  std::multiset<std::string> visited;
  bool grown = false;
  Cache::ApplyToAllEntriesOptions opts;
  opts.average_entries_per_lock = 4;
  cache->ApplyToAllEntries(
      [&](const Slice& key, Cache::ObjectPtr /*value*/, size_t /*charge*/,
          const Cache::CacheItemHelper* /*helper*/) {
        visited.insert(key.ToString());
        if (!grown && visited.size() == 3) {
          grown = true;
          for (uint32_t i = 100; i < 200; ++i) {
            EXPECT_OK(Insert(cache.get(), K(i)));
          }
          EXPECT_GT(cache->GetTableAddressCount(), length_before);
        }
      },
      opts);
  ASSERT_TRUE(grown);
  for (uint32_t i = 0; i < 15; ++i) {
    EXPECT_EQ(1, visited.count(K(i))) << i;
  }
  for (const std::string& key : visited) {
    EXPECT_EQ(1, visited.count(key));
  }
}

// The table grows as entries arrive, and every key, of whatever length,
// stays findable through growth and erasure.
TEST_F(FIFOCacheTest, TableGrowthKeepsEveryKeyFindable) {
  // Room in the table for every key.
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 8 << 20);
  const size_t initial_slots = cache->GetTableAddressCount();
  constexpr uint32_t kKeys = 5000;
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_OK(Insert(cache.get(), L(i)));
  }
  EXPECT_GT(cache->GetTableAddressCount(), initial_slots);
  EXPECT_EQ(kKeys, cache->GetOccupancyCount());
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_TRUE(Contains(cache.get(), L(i))) << i;
  }
  for (uint32_t i = 0; i < kKeys; i += 3) {
    cache->Erase(L(i));
  }
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_EQ(i % 3 != 0, Contains(cache.get(), L(i))) << i;
  }
}

// Inserts, lookups, erases, releases that erase, and capacity changes from
// several threads, with keys of several lengths: every handle reads its own
// live value, and each value and HeapKey is freed exactly once.
TEST_F(FIFOCacheTest, ConcurrentStressKeepsLifetimeExact) {
  const int64_t heap_keys = LiveHeapKeys();
  {
    std::shared_ptr<Cache> cache = NewCache(
        /*capacity*/ 64 << 10, /*strict*/ false, kFullChargeCacheMetadata,
        /*num_shard_bits*/ 1);
    constexpr int kThreads = 8;
    std::vector<port::Thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&cache, t] {
        Random rnd(7 + t);
        std::vector<std::pair<Cache::Handle*, uint32_t>> held;
        std::vector<std::pair<Cache::Handle*, uint32_t>> long_held;
        for (int i = 0; i < 20000; ++i) {
          const uint32_t k = rnd.Skewed(11);
          switch (rnd.Uniform(12)) {
            case 0:
            case 1:
            case 2: {
              Cache::Handle* h = nullptr;
              Status s = cache->Insert(L(k), new CountedValue(k),
                                       CountedHelper(), 64 + rnd.Uniform(2048),
                                       rnd.OneIn(2) ? &h : nullptr);
              ASSERT_TRUE(s.ok() || s.IsOkOverwritten()) << s.ToString();
              if (h != nullptr) {
                CheckCountedValue(cache.get(), h, k);
                held.emplace_back(h, k);
              }
              break;
            }
            case 3:
              cache->Erase(L(k));
              break;
            case 4:
              if (rnd.OneIn(100)) {
                cache->SetCapacity(rnd.OneIn(2) ? (16 << 10) : (64 << 10));
              }
              break;
            default: {
              Cache::Handle* h = cache->Lookup(L(k));
              if (h == nullptr) {
                break;
              }
              CheckCountedValue(cache.get(), h, k);
              if (rnd.OneIn(50) && long_held.size() < 4) {
                long_held.emplace_back(h, k);
              } else {
                held.emplace_back(h, k);
              }
              break;
            }
          }
          if (held.size() > 4 || (!held.empty() && rnd.OneIn(2))) {
            const size_t idx = rnd.Uniform(static_cast<int>(held.size()));
            CheckCountedValue(cache.get(), held[idx].first, held[idx].second);
            cache->Release(held[idx].first,
                           /*erase_if_last_ref*/ rnd.OneIn(10));
            held[idx] = held.back();
            held.pop_back();
          }
          if (!long_held.empty() && rnd.OneIn(2000)) {
            CheckCountedValue(cache.get(), long_held.back().first,
                              long_held.back().second);
            cache->Release(long_held.back().first);
            long_held.pop_back();
          }
        }
        for (auto* list : {&held, &long_held}) {
          for (const auto& [h, k] : *list) {
            CheckCountedValue(cache.get(), h, k);
            cache->Release(h);
          }
        }
      });
    }
    for (port::Thread& thread : threads) {
      thread.join();
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
    size_t charge_sum = 0;
    cache->ApplyToAllEntries(
        [&](const Slice& /*key*/, Cache::ObjectPtr value, size_t charge,
            const Cache::CacheItemHelper* /*helper*/) {
          EXPECT_EQ(kLiveMagic,
                    static_cast<CountedValue*>(value)->magic.load());
          charge_sum += charge;
        },
        {});
    // An erased entry whose last reference went away in a race can stay
    // charged, and out of ApplyToAllEntries, until eviction or
    // EraseUnRefEntries frees it. Afterwards only the table's metadata is
    // charged.
    const size_t metadata = Fifo(cache.get())->TEST_GetMetadataCharge();
    EXPECT_GE(metadata, cache->GetTableAddressCount() * kSlotCharge);
    EXPECT_LE(charge_sum + metadata, cache->GetUsage());
    cache->EraseUnRefEntries();
    EXPECT_EQ(0, cache->GetOccupancyCount());
    EXPECT_EQ(metadata, cache->GetUsage());
    EXPECT_EQ(0, Fifo(cache.get())->TEST_GetProbationUsage());
  }
  EXPECT_EQ(0, LiveValues().load());
  EXPECT_TRUE(HeapKeysAbove(heap_keys, 0));
}

// Under strict_capacity_limit, concurrent inserts (with and without a
// handle), hits and releases never take usage above capacity, even while
// pins hold most of it.
TEST_F(FIFOCacheTest, StrictCapacityHoldsUnderConcurrency) {
  static constexpr size_t kCapacity = 32 << 10;
  std::shared_ptr<Cache> cache =
      NewCache(kCapacity, /*strict*/ true, kDontChargeCacheMetadata);
  std::vector<port::Thread> threads;
  threads.reserve(6);
  for (int t = 0; t < 6; ++t) {
    threads.emplace_back([&cache, t] {
      Random rnd(50 + t);
      std::vector<Cache::Handle*> held;
      for (int i = 0; i < 20000; ++i) {
        const std::string key = K(rnd.Skewed(10));
        if (rnd.OneIn(3)) {
          Cache::Handle* h = nullptr;
          Status s = Insert(cache.get(), key, 512 + rnd.Uniform(2048),
                            rnd.OneIn(2) ? &h : nullptr);
          EXPECT_TRUE(s.ok() || s.IsOkOverwritten() || s.IsMemoryLimit())
              << s.ToString();
          if (h != nullptr) {
            held.push_back(h);
          }
        } else if (Cache::Handle* h = cache->Lookup(key)) {
          held.push_back(h);
        }
        // Up to six pins per thread hold more than the capacity in total,
        // so inserts keep running into the limit.
        if (held.size() > 6 || (!held.empty() && rnd.OneIn(4))) {
          cache->Release(held.front(), rnd.OneIn(10));
          held.erase(held.begin());
        }
        ASSERT_LE(cache->GetUsage(), kCapacity);
      }
      for (Cache::Handle* h : held) {
        cache->Release(h);
      }
    });
  }
  for (port::Thread& thread : threads) {
    thread.join();
  }
  EXPECT_LE(cache->GetUsage(), kCapacity);
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// Probation evicts in insertion order: with nothing hit twice, the oldest
// entry goes first.
TEST_F(FIFOCacheTest, FifoOrderEvictsOldest) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 3);
  RecordEvictions(cache.get(), &evicted);
  for (const char* k : {"a", "b", "c", "d"}) {
    ASSERT_OK(Insert(cache.get(), K(k)));
  }
  EXPECT_EQ(std::vector<std::string>{K("a")}, evicted);
  EXPECT_EQ(nullptr, cache->Lookup(K("a")));
  for (const char* k : {"b", "c", "d"}) {
    EXPECT_TRUE(Contains(cache.get(), K(k))) << k;
  }
  // One hit each promotes none of them, so b, the oldest, goes next.
  ASSERT_OK(Insert(cache.get(), K("e")));
  EXPECT_EQ((std::vector<std::string>{K("a"), K("b")}), evicted);
  EXPECT_EQ((std::set<std::string>{K("c"), K("d"), K("e")}), Keys(cache.get()));
  EXPECT_EQ(3, cache->GetUsage());
  EXPECT_EQ(3, cache->GetOccupancyCount());
  EXPECT_EQ(3, Fifo(cache.get())->TEST_GetProbationUsage());
}

// A hit leaves an entry where it was in the queue, and so does holding it
// while others are evicted: a, hit once, is still the oldest and goes
// first, then b, held across that eviction.
TEST_F(FIFOCacheTest, HitsDoNotReorder) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 3);
  RecordEvictions(cache.get(), &evicted);
  for (const char* k : {"a", "b", "c"}) {
    ASSERT_OK(Insert(cache.get(), K(k)));
  }
  EXPECT_TRUE(Contains(cache.get(), K("a")));
  Cache::Handle* hb = cache->Lookup(K("b"));
  ASSERT_NE(nullptr, hb);
  ASSERT_OK(Insert(cache.get(), K("d")));
  EXPECT_EQ(std::vector<std::string>{K("a")}, evicted);
  EXPECT_FALSE(cache->Release(hb));
  ASSERT_OK(Insert(cache.get(), K("e")));
  EXPECT_EQ((std::vector<std::string>{K("a"), K("b")}), evicted);
}

// A pinned head moves behind the scan, so the next eviction starts at an
// unpinned entry instead of meeting the pins again, and once released the
// pins go from their new places. Heads sent to the resident queue instead
// would outlive d; heads left in place would go before c2.
TEST_F(FIFOCacheTest, PinnedEntriesMoveBehindTheScan) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 5);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  Cache::Handle* p1 = nullptr;
  Cache::Handle* p2 = nullptr;
  ASSERT_OK(Insert(cache.get(), K("p1"), 1, &p1));
  ASSERT_OK(Insert(cache.get(), K("p2"), 1, &p2));
  for (const char* k : {"c0", "c1", "c2", "d"}) {
    ASSERT_OK(Insert(cache.get(), K(k)));
  }
  EXPECT_EQ(std::vector<std::string>{K("c0")}, evicted);
  // Seen pinned, so out of probation's share.
  EXPECT_EQ(2, fifo->TEST_GetProbationPinnedUsage());
  ASSERT_OK(Insert(cache.get(), K("e")));
  EXPECT_EQ((std::vector<std::string>{K("c0"), K("c1")}), evicted);
  EXPECT_FALSE(cache->Release(p1));
  EXPECT_FALSE(cache->Release(p2));
  for (const char* k : {"f", "g", "h", "i"}) {
    ASSERT_OK(Insert(cache.get(), K(k)));
  }
  EXPECT_EQ((std::vector<std::string>{K("c0"), K("c1"), K("c2"), K("p1"),
                                      K("p2"), K("d")}),
            evicted);
  EXPECT_EQ(0, fifo->TEST_GetProbationPinnedUsage());
}

// Eviction passes over several pinned heads in a row and takes the
// unpinned entries behind them.
TEST_F(FIFOCacheTest, EvictionSkipsMultiplePinned) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 4);
  RecordEvictions(cache.get(), &evicted);
  for (const char* k : {"a", "b", "c", "d"}) {
    ASSERT_OK(Insert(cache.get(), K(k)));
  }
  Cache::Handle* pa = cache->Lookup(K("a"));
  Cache::Handle* pb = cache->Lookup(K("b"));
  ASSERT_NE(nullptr, pa);
  ASSERT_NE(nullptr, pb);
  ASSERT_OK(Insert(cache.get(), K("e")));
  ASSERT_OK(Insert(cache.get(), K("f")));
  EXPECT_EQ((std::vector<std::string>{K("c"), K("d")}), evicted);
  EXPECT_EQ((std::set<std::string>{K("a"), K("b"), K("e"), K("f")}),
            Keys(cache.get()));
  EXPECT_FALSE(cache->Release(pa));
  EXPECT_FALSE(cache->Release(pb));
}

// One large head covers the whole deficit.
TEST_F(FIFOCacheTest, MixedChargesSingleVictim) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  ASSERT_OK(Insert(cache.get(), K("a"), 6));
  ASSERT_OK(Insert(cache.get(), K("b"), 2));
  ASSERT_OK(Insert(cache.get(), K("c"), 2));
  ASSERT_OK(Insert(cache.get(), K("d"), 5));
  EXPECT_EQ(std::vector<std::string>{K("a")}, evicted);
  EXPECT_EQ(9, cache->GetUsage());
}

// Eviction takes as many heads as the deficit needs and no more: e's 6 need
// 4 freed, so a and b go and c stays. (HyperClockCache frees the incoming
// charge in full, which would take c too.)
TEST_F(FIFOCacheTest, MixedChargesMultiVictim) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  for (const char* k : {"a", "b", "c", "d"}) {
    ASSERT_OK(Insert(cache.get(), K(k), 2));
  }
  ASSERT_OK(Insert(cache.get(), K("e"), 6));
  EXPECT_EQ((std::vector<std::string>{K("a"), K("b")}), evicted);
  EXPECT_EQ(10, cache->GetUsage());
}

// Two hits promote and one does not: a, hit twice, outlives the pressure
// that evicts b, hit once, and c, inserted with a handle and hit once after
// (the reference Insert returns is not a hit). Promoting at one hit would
// keep b, at three or never would evict a, and plain FIFO would evict a
// first.
TEST_F(FIFOCacheTest, TwoHitsPromoteOneHitEvicts) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  ASSERT_OK(Insert(cache.get(), K("a")));
  HitNTimes(cache.get(), K("a"), 2);
  ASSERT_OK(Insert(cache.get(), K("b")));
  HitNTimes(cache.get(), K("b"), 1);
  Cache::Handle* hc = nullptr;
  ASSERT_OK(Insert(cache.get(), K("c"), 1, &hc));
  ASSERT_NE(nullptr, hc);
  EXPECT_FALSE(cache->Release(hc));
  HitNTimes(cache.get(), K("c"), 1);
  for (uint32_t i = 0; i < 7; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_OK(Insert(cache.get(), K("x")));
  EXPECT_EQ(std::vector<std::string>{K("b")}, evicted);
  ASSERT_OK(Insert(cache.get(), K("y")));
  EXPECT_EQ((std::vector<std::string>{K("b"), K("c")}), evicted);
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_EQ(10, cache->GetUsage());
  EXPECT_EQ(9, Fifo(cache.get())->TEST_GetProbationUsage());
}

// The oldest entry, hit twice, outlives pressure that evicts newer cold
// ones. Plain FIFO, or never promoting, would evict it first.
TEST_F(FIFOCacheTest, PromotedEntrySurvivesColdPressure) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  ASSERT_OK(Insert(cache.get(), K("old")));
  HitNTimes(cache.get(), K("old"), 2);
  for (uint32_t i = 0; i < 9; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_OK(Insert(cache.get(), K("new")));
  EXPECT_EQ(std::vector<std::string>{K(0)}, evicted);
  EXPECT_TRUE(Holds(cache.get(), K("old")));
  EXPECT_EQ(10, cache->GetUsage());
}

// The resident queue is a clock over the table: each time its hand passes
// an unpinned entry it takes one hit off, keeping at most three, and it
// evicts an entry with none left. So entries promoted together and then hit
// 0, 1, 2 and 3 times go in that order, wherever their slots are, and one
// hit 100 times lasts no longer than one hit 3 times. Probation holds only
// pinned entries here, which leave its share once a step sees them, so
// each insert over capacity takes from the resident clock.
TEST_F(FIFOCacheTest, ResidentClockCountsHitsUpToThree) {
  std::vector<std::string> evicted;
  std::vector<Cache::Handle*> pins;
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
    RecordEvictions(cache.get(), &evicted);
    const std::vector<std::string> k = {K("k0"), K("k1"), K("k2"), K("k3"),
                                        K("k100")};
    for (const std::string& key : k) {
      ASSERT_OK(Insert(cache.get(), key));
      HitNTimes(cache.get(), key, 2);
    }
    for (uint32_t i = 0; i < 5; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i)));
    }
    // Promotes the five, with their hits reset, and evicts K(0).
    ASSERT_OK(Insert(cache.get(), K("t")));
    ASSERT_EQ(std::vector<std::string>{K(0)}, evicted);
    for (uint32_t i = 1; i < 5; ++i) {
      cache->Erase(K(i));
    }
    cache->Erase(K("t"));
    ASSERT_EQ(0, Fifo(cache.get())->TEST_GetProbationUsage());
    HitNTimes(cache.get(), k[1], 1);
    HitNTimes(cache.get(), k[2], 2);
    HitNTimes(cache.get(), k[3], 3);
    HitNTimes(cache.get(), k[4], 100);
    // Five pinned inserts fill the cache, and each of the next five evicts
    // from the resident queue.
    for (uint32_t i = 100; i < 110; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
      pins.push_back(h);
    }
    ASSERT_EQ(6, evicted.size());
    EXPECT_EQ(k[0], evicted[1]);
    EXPECT_EQ(k[1], evicted[2]);
    EXPECT_EQ(k[2], evicted[3]);
    EXPECT_EQ((std::set<std::string>{k[3], k[4]}),
              (std::set<std::string>{evicted[4], evicted[5]}));
    for (Cache::Handle* h : pins) {
      cache->Release(h);
    }
  }

  // Alone in the resident queue, an entry hit 100 times goes within one
  // insert, whose eviction gives the clock four laps (one more than the
  // most hits it keeps).
  evicted.clear();
  pins.clear();
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  ASSERT_OK(Insert(cache.get(), K("m")));
  HitNTimes(cache.get(), K("m"), 2);
  for (uint32_t i = 0; i < 9; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_OK(Insert(cache.get(), K("t")));
  ASSERT_EQ(std::vector<std::string>{K(0)}, evicted);
  for (uint32_t i = 1; i < 9; ++i) {
    cache->Erase(K(i));
  }
  cache->Erase(K("t"));
  HitNTimes(cache.get(), K("m"), 100);
  for (uint32_t i = 100; i < 110; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
    pins.push_back(h);
  }
  EXPECT_EQ((std::vector<std::string>{K(0), K("m")}), evicted);
  for (Cache::Handle* h : pins) {
    cache->Release(h);
  }
}

// Promotion moves an entry's charge out of probation without counting it
// again or dropping it, and leaves pinned usage alone.
TEST_F(FIFOCacheTest, PromotionPreservesUsageAndPinnedUsage) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 30);
  FIFOCache* fifo = Fifo(cache.get());
  Cache::Handle* pinned = nullptr;
  ASSERT_OK(Insert(cache.get(), K("pin"), 4, &pinned));
  ASSERT_OK(Insert(cache.get(), K("k"), 10));
  HitNTimes(cache.get(), K("k"), 2);
  ASSERT_OK(Insert(cache.get(), K("a"), 9));
  ASSERT_OK(Insert(cache.get(), K("b"), 7));
  EXPECT_EQ(30, cache->GetUsage());
  EXPECT_EQ(4, cache->GetPinnedUsage());
  EXPECT_EQ(30, fifo->TEST_GetProbationUsage());
  // The pinned head rotates, k is promoted and a evicted: usage drops by a's
  // charge only, and probation's by k's and a's, gaining d's.
  ASSERT_OK(Insert(cache.get(), K("d"), 5));
  EXPECT_TRUE(Holds(cache.get(), K("k")));
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_EQ(26, cache->GetUsage());
  EXPECT_EQ(4, cache->GetPinnedUsage());
  EXPECT_EQ(16, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(4, fifo->TEST_GetProbationPinnedUsage());
  EXPECT_FALSE(cache->Release(pinned));
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// Priority is ignored: HIGH and LOW entries with the same hits go the same
// way. The first half passes on plain FIFO too; the second needs promotion.
TEST_F(FIFOCacheTest, PriorityIgnoredForPlacement) {
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 6);
    ASSERT_OK(Insert(cache.get(), K("h"), 1, nullptr, Cache::Priority::HIGH));
    ASSERT_OK(Insert(cache.get(), K("l"), 1, nullptr, Cache::Priority::LOW));
    for (uint32_t i = 0; i < 4; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i)));
    }
    ASSERT_OK(Insert(cache.get(), K("x")));
    EXPECT_FALSE(Holds(cache.get(), K("h")));
    EXPECT_TRUE(Holds(cache.get(), K("l")));
  }
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 6);
  ASSERT_OK(Insert(cache.get(), K("h2"), 1, nullptr, Cache::Priority::HIGH));
  HitNTimes(cache.get(), K("h2"), 2);
  ASSERT_OK(Insert(cache.get(), K("l2"), 1, nullptr, Cache::Priority::LOW));
  HitNTimes(cache.get(), K("l2"), 2);
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_OK(Insert(cache.get(), K("y")));
  EXPECT_TRUE(Holds(cache.get(), K("h2")));
  EXPECT_TRUE(Holds(cache.get(), K("l2")));
  EXPECT_FALSE(Holds(cache.get(), K(0)));
}

// With every entry in both queues pinned, eviction gives up instead of
// spinning on pinned heads, and a non-strict insert goes over capacity, as
// in HyperClockCache. The next insert evicts the unpinned entry the last one
// left, and once the pins go, inserts evict back to capacity.
TEST_F(FIFOCacheTest, FullyPinnedTerminates) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  ASSERT_OK(Insert(cache.get(), K("a")));
  HitNTimes(cache.get(), K("a"), 2);
  ASSERT_OK(Insert(cache.get(), K("b")));
  HitNTimes(cache.get(), K("b"), 2);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  // Promotes a and b and evicts K(0), filling both queues.
  ASSERT_OK(Insert(cache.get(), K("g")));
  EXPECT_EQ(10, cache->GetOccupancyCount());
  EXPECT_FALSE(Holds(cache.get(), K(0)));
  std::vector<Cache::Handle*> pinned;
  for (const std::string& key : Keys(cache.get())) {
    pinned.push_back(cache->Lookup(key));
    ASSERT_NE(nullptr, pinned.back());
  }
  ASSERT_OK(Insert(cache.get(), K("h")));
  EXPECT_TRUE(Holds(cache.get(), K("h")));
  EXPECT_EQ(11, cache->GetUsage());
  Cache::Handle* hi = nullptr;
  ASSERT_OK(Insert(cache.get(), K("i"), 1, &hi));
  ASSERT_NE(nullptr, hi);
  EXPECT_FALSE(Holds(cache.get(), K("h")));
  EXPECT_EQ(11, cache->GetUsage());
  EXPECT_EQ(11, cache->GetPinnedUsage());
  pinned.push_back(hi);
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
  EXPECT_EQ(0, cache->GetPinnedUsage());
  ASSERT_OK(Insert(cache.get(), K("j")));
  EXPECT_EQ(10, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_TRUE(Holds(cache.get(), K("b")));
}

// Probation over its share evicts nothing while the cache has room; under
// pressure its oldest entry goes first, ahead of a later one hit twice.
TEST_F(FIFOCacheTest, ProbationOverBudgetEvictsProbationHead) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  EXPECT_EQ(12 * kU, fifo->TEST_GetProbationTarget());
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  ASSERT_OK(Insert(cache.get(), K("b"), kU));
  HitNTimes(cache.get(), K("b"), 2);
  for (uint32_t i = 0; i < 11; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_EQ(13 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(13 * kU, cache->GetUsage());
  for (uint32_t i = 100; i < 187; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_EQ(100 * kU, cache->GetUsage());
  EXPECT_TRUE(evicted.empty());
  ASSERT_OK(Insert(cache.get(), K("e"), kU));
  EXPECT_EQ(std::vector<std::string>{K("a")}, evicted);
  EXPECT_TRUE(Holds(cache.get(), K("b")));
  EXPECT_EQ(100 * kU, cache->GetUsage());
}

// With probation within its share, pressure draws on the resident queue.
// A probation sweep that promotes keeps going until it evicts, even once
// the promotions take probation under its share, so the entries it just
// promoted (with no hits yet) are not what the insert evicts.
TEST_F(FIFOCacheTest, ProbationUnderBudgetDrawsFromResident) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  std::set<std::string> resident;
  for (uint32_t i = 0; i < 89; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
    HitNTimes(cache.get(), K(i), 2);
    resident.insert(K(i));
  }
  ASSERT_OK(Insert(cache.get(), K("s"), kU));
  for (uint32_t i = 100; i < 110; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_EQ(100 * kU, cache->GetUsage());
  // Promotes K(0)..K(88) and evicts s.
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  EXPECT_EQ(std::vector<std::string>{K("s")}, evicted);
  // Probation holds K(100)..K(109) and t, so x lands it at its share, not
  // over: the resident clock pays.
  EXPECT_EQ(11 * kU, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_GE(evicted.size(), 2);
  for (size_t i = 1; i < evicted.size(); ++i) {
    EXPECT_EQ(1, resident.count(evicted[i])) << i;
  }
  for (uint32_t i = 100; i < 110; ++i) {
    EXPECT_TRUE(Holds(cache.get(), K(i))) << i;
  }
  EXPECT_TRUE(Holds(cache.get(), K("t")));
  EXPECT_TRUE(Holds(cache.get(), K("x")));
  EXPECT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_LE(cache->GetUsage(), 100 * kU);
}

// The queue choice counts the incoming entry, which joins probation: at its
// share, probation pays when the insert would take it over, and the
// resident queue pays when it lands exactly at the share.
TEST_F(FIFOCacheTest, ProbationBudgetCountsIncomingCharge) {
  {
    std::vector<std::string> evicted;
    std::shared_ptr<Cache> cache = NewCache(100 * kU);
    RecordEvictions(cache.get(), &evicted);
    FIFOCache* fifo = Fifo(cache.get());
    ASSERT_OK(Insert(cache.get(), K("big"), 88 * kU));
    HitNTimes(cache.get(), K("big"), 2);
    for (uint32_t i = 0; i < 12; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), kU));
    }
    // Promotes big, evicts K(0).
    ASSERT_OK(Insert(cache.get(), K("t"), kU));
    ASSERT_EQ(std::vector<std::string>{K(0)}, evicted);
    EXPECT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
    // An incoming 5 would take probation to 17: K(1)..K(5) go.
    ASSERT_OK(Insert(cache.get(), K("x"), 5 * kU));
    EXPECT_EQ((std::vector<std::string>{K(0), K(1), K(2), K(3), K(4), K(5)}),
              evicted);
    EXPECT_TRUE(Holds(cache.get(), K("big")));
    EXPECT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
    EXPECT_EQ(100 * kU, cache->GetUsage());
  }
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 89 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  for (uint32_t i = 0; i < 11; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_EQ(std::vector<std::string>{K(0)}, evicted);
  EXPECT_EQ(11 * kU, fifo->TEST_GetProbationUsage());
  // 11 plus the incoming 1 lands exactly at the share.
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  EXPECT_EQ((std::vector<std::string>{K(0), K("big")}), evicted);
}

// Long-lived pins in probation, such as cache reservations or pinned index
// and filter blocks, leave its share once an eviction step sees them
// pinned, so with 13 pins over a share of 12 the resident clock still runs:
// an entry there with no hits goes rather than the next unpinned probation
// entry.
TEST_F(FIFOCacheTest, PinnedProbationEntriesDoNotHoldBudget) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 200; i < 213; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), kU, &h));
    pins.push_back(h);
  }
  std::set<std::string> hot;
  for (uint32_t i = 0; i < 80; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
    HitNTimes(cache.get(), K(i), 2);
    hot.insert(K(i));
  }
  for (uint32_t i = 100; i < 107; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_EQ(100 * kU, cache->GetUsage());
  // Over its share: the pins rotate, the 80 hot entries are promoted and
  // K(100) is evicted.
  ASSERT_OK(Insert(cache.get(), K("x0"), kU));
  ASSERT_EQ(std::vector<std::string>{K(100)}, evicted);
  // Probation holds K(101)..K(106), x0 and the 13 pins.
  EXPECT_EQ(20 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(13 * kU, fifo->TEST_GetProbationPinnedUsage());
  ASSERT_OK(Insert(cache.get(), K("x1"), kU));
  ASSERT_GE(evicted.size(), 2);
  for (size_t i = 1; i < evicted.size(); ++i) {
    EXPECT_EQ(1, hot.count(evicted[i])) << i;
  }
  EXPECT_TRUE(Holds(cache.get(), K(101)));
  EXPECT_TRUE(Holds(cache.get(), K("x1")));
  for (Cache::Handle* h : pins) {
    EXPECT_FALSE(cache->Release(h));
  }
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// The resident clock, passing probation entries, notes whether they are
// pinned, so pins released since a probation step saw them come back into
// probation's share at once. Here the clock's pass over them hands the
// insert back to probation, which evicts its head, where the stale view
// (10 of the share's 12 unpinned) would leave the clock to evict a resident
// entry.
TEST_F(FIFOCacheTest, ResidentClockTracksProbationPins) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 200; i < 220; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), kU, &h));
    pins.push_back(h);
  }
  for (uint32_t i = 0; i < 70; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
    HitNTimes(cache.get(), K(i), 2);
  }
  for (uint32_t i = 100; i < 110; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // The pins rotate, the 70 hot entries are promoted and K(100) is evicted.
  ASSERT_OK(Insert(cache.get(), K("x0"), kU));
  ASSERT_EQ(std::vector<std::string>{K(100)}, evicted);
  ASSERT_EQ(20 * kU, fifo->TEST_GetProbationPinnedUsage());
  for (Cache::Handle* h : pins) {
    EXPECT_FALSE(cache->Release(h));
  }
  // One hit each, so the clock evicts nothing in its first lap.
  for (uint32_t i = 0; i < 70; ++i) {
    HitNTimes(cache.get(), K(i), 1);
  }
  ASSERT_OK(Insert(cache.get(), K("x1"), kU));
  EXPECT_EQ((std::vector<std::string>{K(100), K(101)}), evicted);
  EXPECT_LT(fifo->TEST_GetProbationPinnedUsage(), 20 * kU);
}

// Probation's usage follows inserts, evictions, promotions, erases and
// replacing inserts by charge, not by entry count.
TEST_F(FIFOCacheTest, ProbationUsageAccounting) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  EXPECT_EQ(0, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K("a"), 5 * kU));
  ASSERT_OK(Insert(cache.get(), K("b"), 7 * kU));
  EXPECT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  HitNTimes(cache.get(), K("a"), 2);
  for (uint32_t i = 0; i < 88; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_EQ(100 * kU, cache->GetUsage());
  EXPECT_EQ(100 * kU, fifo->TEST_GetProbationUsage());
  // a is promoted (5 leave probation), b evicted (7 leave), t joins (1).
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_FALSE(Holds(cache.get(), K("b")));
  EXPECT_EQ(89 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(94 * kU, cache->GetUsage());
  // Erasing from the resident queue leaves probation alone; erasing from
  // probation takes from it.
  cache->Erase(K("a"));
  EXPECT_EQ(89 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(89 * kU, cache->GetUsage());
  cache->Erase(K(0));
  EXPECT_EQ(88 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(88 * kU, cache->GetUsage());
  // Replacing a probation entry swaps its charge.
  Status s = Insert(cache.get(), K(1), 3 * kU);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  EXPECT_EQ(90 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(90 * kU, cache->GetUsage());
}

// Probation's share is 12% of capacity, rounded down: exact on small
// capacities, zero on tiny ones (so eviction always starts in probation),
// following SetCapacity, and without overflow on the largest.
TEST_F(FIFOCacheTest, ProbationTargetArithmetic) {
  EXPECT_EQ(12, Fifo(NewCache(100).get())->TEST_GetProbationTarget());
  EXPECT_EQ(11, Fifo(NewCache(99).get())->TEST_GetProbationTarget());
  EXPECT_EQ(0, Fifo(NewCache(8).get())->TEST_GetProbationTarget());
  std::shared_ptr<Cache> cache = NewCache(100);
  FIFOCache* fifo = Fifo(cache.get());
  cache->SetCapacity(200);
  EXPECT_EQ(24, fifo->TEST_GetProbationTarget());
  cache->SetCapacity(50);
  EXPECT_EQ(6, fifo->TEST_GetProbationTarget());
  // SIZE_MAX is 100q + 15, so 12% of it is 12q + 1, where capacity * 12 / 100
  // would wrap. Raised by SetCapacity, so the test doesn't reserve a table
  // for SIZE_MAX.
  cache->SetCapacity(SIZE_MAX);
  const size_t q = SIZE_MAX / 100;
  EXPECT_EQ(12 * q + 1, fifo->TEST_GetProbationTarget());
}

// Each shard's table reserves max(2048, capacity / 64) slots, as FIFOCache
// did before this table, so at its load factor of 0.6 it holds about one
// entry per 107 bytes of capacity, where HyperClockCache's holds one per
// 450. Past that, new keys are evicted by occupancy.
TEST_F(FIFOCacheTest, TableHoldsOneEntryPer107Bytes) {
  for (const auto& [capacity, slots, entries] :
       {std::tuple<size_t, size_t, size_t>{1 << 20, 16384, 9831},
        {64 << 10, 2048, 1229}}) {
    SCOPED_TRACE(capacity);
    std::shared_ptr<Cache> cache = NewCache(capacity);
    for (uint32_t i = 0; i < 2 * entries; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i)));
    }
    EXPECT_EQ(slots, cache->GetTableAddressCount());
    EXPECT_EQ(entries, cache->GetOccupancyCount());
    EXPECT_EQ(entries, cache->GetUsage());
  }
}

// max_capacity reserves the table for a capacity SetCapacity raises the
// cache to later, so the table can hold more entries than the construction
// capacity's reservation (2048 slots, 1229 entries, for 64 KiB), split
// across shards like capacity. A table at its reservation charges nothing
// for the growth it can't make.
TEST_F(FIFOCacheTest, MaxCapacityReservesForLaterGrowth) {
  constexpr uint32_t kEntries = 5000;
  for (CacheMetadataChargePolicy policy :
       {kDontChargeCacheMetadata, kFullChargeCacheMetadata}) {
    for (int num_shard_bits : {0, 1}) {
      for (size_t max_capacity : {size_t{0}, size_t{1} << 20}) {
        SCOPED_TRACE(std::to_string(policy) + " " +
                     std::to_string(num_shard_bits) + " " +
                     std::to_string(max_capacity));
        std::shared_ptr<Cache> cache =
            NewCache(/*capacity*/ 64 << 10, /*strict*/ false, policy,
                     num_shard_bits, max_capacity);
        cache->SetCapacity(1 << 20);
        for (uint32_t i = 0; i < kEntries; ++i) {
          ASSERT_OK(Insert(cache.get(), K(i)));
        }
        if (max_capacity == 0) {
          EXPECT_EQ(cache->GetOccupancyCount(), (1229 << num_shard_bits));
          EXPECT_EQ(cache->GetTableAddressCount(), (2048 << num_shard_bits));
        } else {
          EXPECT_EQ(kEntries, cache->GetOccupancyCount());
          EXPECT_EQ(kEntries, Keys(cache.get()).size());
        }
        const size_t metadata =
            policy == kFullChargeCacheMetadata
                ? Fifo(cache.get())->TEST_GetMetadataCharge()
                : 0;
        EXPECT_EQ(cache->GetOccupancyCount() + metadata, cache->GetUsage());
      }
    }
  }
}

// Inserting a cached key replaces the entry and returns OkOverwritten: the
// new value and charge are what Lookup finds and the old value is freed.
// The new entry starts in probation with no hits, where HyperClockCache
// keeps the old entry, counts each duplicate insert as a hit on it and
// drops the new value, so here a key inserted three times is still evicted
// from probation.
TEST_F(FIFOCacheTest, InsertReplacesUnpinnedEntry) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  ASSERT_OK(
      cache->Insert(K("a"), new CountedValue(1), CountedHelper(), 2 * kU));
  for (uint32_t v : {2, 3}) {
    Status s = cache->Insert(K("a"), new CountedValue(v), CountedHelper(),
                             (v + 2) * kU);
    EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  }
  EXPECT_EQ(1, LiveValues().load());
  EXPECT_EQ(5 * kU, cache->GetUsage());
  EXPECT_EQ(1, cache->GetOccupancyCount());
  EXPECT_EQ(5 * kU, Fifo(cache.get())->TEST_GetProbationUsage());
  EXPECT_EQ((std::set<std::string>{K("a")}), Keys(cache.get()));
  for (uint32_t i = 0; i < 6; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_EQ(0, LiveValues().load());
  ASSERT_OK(cache->Insert(K("b"), new CountedValue(4), CountedHelper(), kU));
  Status s = cache->Insert(K("b"), new CountedValue(5), CountedHelper(), kU);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  Cache::Handle* h = cache->Lookup(K("b"));
  ASSERT_NE(nullptr, h);
  CheckCountedValue(cache.get(), h, 5);
  EXPECT_EQ(kU, cache->GetCharge(h));
  cache->Release(h);
  cache.reset();
  EXPECT_EQ(0, LiveValues().load());
}

// A replaced entry that is pinned stays valid, charged and pinned until its
// last release, which frees it, and it keeps its slot until then (today's
// FIFOCache took it out of the table at once), while Lookup finds the new
// one.
TEST_F(FIFOCacheTest, InsertReplacingPinnedEntryKeepsItUntilRelease) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  Cache::Handle* old = nullptr;
  ASSERT_OK(cache->Insert(K("a"), new CountedValue(1), CountedHelper(), 2 * kU,
                          &old));
  Status s =
      cache->Insert(K("a"), new CountedValue(2), CountedHelper(), 5 * kU);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  EXPECT_EQ(7 * kU, cache->GetUsage());
  EXPECT_EQ(2 * kU, cache->GetPinnedUsage());
  EXPECT_EQ(2, cache->GetOccupancyCount());
  EXPECT_EQ(2, LiveValues().load());
  CheckCountedValue(cache.get(), old, 1);
  Cache::Handle* cur = cache->Lookup(K("a"));
  ASSERT_NE(nullptr, cur);
  CheckCountedValue(cache.get(), cur, 2);
  EXPECT_EQ(7 * kU, cache->GetPinnedUsage());
  EXPECT_EQ((std::set<std::string>{K("a")}), Keys(cache.get()));
  EXPECT_TRUE(cache->Release(old));
  EXPECT_EQ(1, LiveValues().load());
  EXPECT_EQ(5 * kU, cache->GetUsage());
  EXPECT_EQ(1, cache->GetOccupancyCount());
  EXPECT_FALSE(cache->Release(cur));
  EXPECT_EQ(0, cache->GetPinnedUsage());
  cache.reset();
  EXPECT_EQ(0, LiveValues().load());
}

// At the table's occupancy limit (1229 entries in 2048 slots) a replacing
// insert still evicts an entry for its slot, since it doesn't look for the
// old entry first. Where that eviction takes the entry being replaced (the
// oldest in probation), the insert returns OK; otherwise it then replaces
// the old entry, leaving one entry fewer. Today's FIFOCache kept occupancy
// and never evicted the entry it replaced. Either way the new value is the
// one found and each value is freed once.
TEST_F(FIFOCacheTest, InsertReplacingAtOccupancyLimit) {
  for (bool victim_oldest : {true, false}) {
    SCOPED_TRACE(victim_oldest);
    std::shared_ptr<Cache> cache = NewCache(64 << 10);
    if (victim_oldest) {
      ASSERT_OK(cache->Insert(K("v"), new CountedValue(1), CountedHelper(), 1));
    }
    for (uint32_t i = 0; i < 1228; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i)));
    }
    if (!victim_oldest) {
      ASSERT_OK(cache->Insert(K("v"), new CountedValue(1), CountedHelper(), 1));
    }
    ASSERT_EQ(1229, cache->GetOccupancyCount());
    Status s = cache->Insert(K("v"), new CountedValue(2), CountedHelper(), 1);
    ASSERT_OK(s);
    EXPECT_EQ(!victim_oldest, s.IsOkOverwritten()) << s.ToString();
    EXPECT_EQ(victim_oldest ? 1229 : 1228, cache->GetOccupancyCount());
    EXPECT_EQ(victim_oldest, Holds(cache.get(), K(0)));
    EXPECT_EQ(1, LiveValues().load());
    Cache::Handle* h = cache->Lookup(K("v"));
    ASSERT_NE(nullptr, h);
    CheckCountedValue(cache.get(), h, 2);
    cache->Release(h);
    cache.reset();
    EXPECT_EQ(0, LiveValues().load());
  }
}

// An insert that finds every entry pinned is still accepted without a slot
// (standalone with a handle, dropped without), so the old value doesn't
// stay visible and the insert returns OkOverwritten. One that is refused
// leaves the old value: a strict insert that can't make room, with a
// handle (MemoryLimit) or without (dropped, OK).
TEST_F(FIFOCacheTest, InsertReplacingWithoutFreeSlotHidesOldValue) {
  for (bool with_handle : {false, true}) {
    SCOPED_TRACE(with_handle);
    std::shared_ptr<Cache> cache = NewCache(64 << 10);
    std::vector<Cache::Handle*> pins;
    for (uint32_t i = 0; i < 1228; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
      pins.push_back(h);
    }
    Cache::Handle* old = nullptr;
    ASSERT_OK(
        cache->Insert(K("k"), new CountedValue(1), CountedHelper(), 1, &old));
    ASSERT_EQ(1229, cache->GetOccupancyCount());

    cache->SetStrictCapacityLimit(true);
    auto* refused = new CountedValue(2);
    Cache::Handle* strict = nullptr;
    Status s = cache->Insert(K("k"), refused, CountedHelper(), 1, &strict);
    EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
    EXPECT_EQ(nullptr, strict);
    DeleteCountedValue(refused, nullptr);
    s = cache->Insert(K("k"), new CountedValue(3), CountedHelper(), 1);
    EXPECT_TRUE(s.ok() && !s.IsOkOverwritten()) << s.ToString();
    Cache::Handle* h = cache->Lookup(K("k"));
    ASSERT_NE(nullptr, h);
    CheckCountedValue(cache.get(), h, 1);
    cache->Release(h);
    cache->SetStrictCapacityLimit(false);

    Cache::Handle* outside = nullptr;
    s = cache->Insert(K("k"), new CountedValue(4), CountedHelper(), 1,
                      with_handle ? &outside : nullptr);
    EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
    EXPECT_EQ(nullptr, cache->Lookup(K("k")));
    EXPECT_EQ(1229, cache->GetOccupancyCount());
    EXPECT_EQ(with_handle ? 2 : 1, LiveValues().load());
    if (with_handle) {
      ASSERT_NE(nullptr, outside);
      CheckCountedValue(cache.get(), outside, 4);
      EXPECT_TRUE(cache->Release(outside));
    }
    CheckCountedValue(cache.get(), old, 1);
    EXPECT_TRUE(cache->Release(old));
    EXPECT_EQ(0, LiveValues().load());
    EXPECT_EQ(1228, cache->GetOccupancyCount());
    for (Cache::Handle* p : pins) {
      EXPECT_FALSE(cache->Release(p));
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
  }
}

// The entry being replaced is found however deep in its chain it is: here
// behind eleven newer entries with the same home, past the ten Lookup's
// first pass reads, so the full walk has to step over the new entry.
TEST_F(FIFOCacheTest, InsertReplacesEntryDeepInItsChain) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  const uint32_t seed = cache->GetHashSeed();
  std::vector<UniqueId64x2> unhashed(13);
  auto same_home = [&](uint64_t i) {
    return Shard::ReverseHash(UniqueId64x2{0xc4a1, (i << 32) | 5}, &unhashed[i],
                              seed)
        .ToString();
  };
  const std::string k = same_home(0);
  ASSERT_OK(cache->Insert(k, new CountedValue(1), CountedHelper(), kU));
  for (uint64_t i = 1; i <= 11; ++i) {
    ASSERT_OK(Insert(cache.get(), same_home(i), kU));
  }
  Status s = cache->Insert(k, new CountedValue(2), CountedHelper(), kU);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  EXPECT_EQ(1, LiveValues().load());
  EXPECT_EQ(12, Keys(cache.get()).size());
  Cache::Handle* h = cache->Lookup(k);
  ASSERT_NE(nullptr, h);
  CheckCountedValue(cache.get(), h, 2);
  cache->Release(h);
  cache.reset();
  EXPECT_EQ(0, LiveValues().load());
}

// At the occupancy limit, inserts replacing hot keys race lookups, erases,
// releases that erase and GetPinnedUsage: every value a holder reads is
// alive and its own, each value is freed once, and afterwards no key has
// two visible entries (Keys fails on a duplicate) and nothing unreferenced
// is left behind.
TEST_F(FIFOCacheTest, ConcurrentReplacesAtOccupancyLimitKeepLifetimeExact) {
  {
    // 2048 slots and 1229 entries a shard, and about one byte an entry, so
    // occupancy, not charge, bounds the table.
    std::shared_ptr<Cache> cache =
        NewCache(256 << 10, /*strict*/ false, kDontChargeCacheMetadata,
                 /*num_shard_bits*/ 1);
    constexpr int kThreads = 8;
    std::atomic<bool> done{false};
    port::Thread poller([&] {
      while (!done.load()) {
        cache->GetPinnedUsage();
      }
    });
    std::vector<port::Thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&cache, t] {
        Random rnd(500 + t);
        std::vector<std::pair<Cache::Handle*, uint32_t>> held;
        for (int i = 0; i < 20000; ++i) {
          const uint32_t k = rnd.Skewed(13);
          const uint32_t op = rnd.Uniform(100);
          if (op < 30) {
            Cache::Handle* h = nullptr;
            Status s =
                cache->Insert(K(k), new CountedValue(k), CountedHelper(),
                              1 + rnd.Uniform(2), rnd.OneIn(2) ? &h : nullptr);
            ASSERT_TRUE(s.ok() || s.IsOkOverwritten()) << s.ToString();
            if (h != nullptr) {
              CheckCountedValue(cache.get(), h, k);
              held.emplace_back(h, k);
            }
          } else if (op < 35) {
            cache->Erase(K(k));
          } else if (Cache::Handle* h = cache->Lookup(K(k))) {
            CheckCountedValue(cache.get(), h, k);
            held.emplace_back(h, k);
          }
          if (held.size() > 8 || (!held.empty() && rnd.OneIn(2))) {
            const size_t idx = rnd.Uniform(static_cast<int>(held.size()));
            CheckCountedValue(cache.get(), held[idx].first, held[idx].second);
            cache->Release(held[idx].first,
                           /*erase_if_last_ref*/ rnd.OneIn(10));
            held[idx] = held.back();
            held.pop_back();
          }
        }
        for (const auto& [h, k] : held) {
          CheckCountedValue(cache.get(), h, k);
          cache->Release(h);
        }
      });
    }
    for (port::Thread& thread : threads) {
      thread.join();
    }
    done.store(true);
    poller.join();
    EXPECT_EQ(0, cache->GetPinnedUsage());
    EXPECT_EQ(2 * 2048, cache->GetTableAddressCount());
    Keys(cache.get());
    // Entries erased while an insert or a lookup held a reference it then
    // dropped are left invisible for eviction to free, as in
    // HyperClockCache, so occupancy can be above the visible count;
    // EraseUnRefEntries frees them all.
    cache->EraseUnRefEntries();
    EXPECT_EQ(0, cache->GetOccupancyCount());
    EXPECT_EQ(0, LiveValues().load());
  }
  EXPECT_EQ(0, LiveValues().load());
}

// Entries erased while pinned keep their slots and their place in
// occupancy until released, so the table grows past them instead of
// refusing new keys while capacity is free, under either limit.
TEST_F(FIFOCacheTest, ErasedPinnedSlotsDoNotBlockGrowth) {
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict);
    std::shared_ptr<Cache> cache = NewCache(1 << 20, strict);
    std::vector<Cache::Handle*> pinned;
    for (uint32_t i = 0; i < 100; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
      pinned.push_back(h);
      cache->Erase(K(i));
    }
    for (uint32_t i = 100; i < 300; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(i), 1, strict ? &h : nullptr));
      if (h != nullptr) {
        cache->Release(h);
      }
    }
    for (uint32_t i = 100; i < 300; ++i) {
      EXPECT_TRUE(Holds(cache.get(), K(i))) << i;
    }
    EXPECT_EQ(300, cache->GetOccupancyCount());
    for (Cache::Handle* h : pinned) {
      EXPECT_TRUE(cache->Release(h));
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
    EXPECT_EQ(200, cache->GetOccupancyCount());
  }
}

// Without strict_capacity_limit, an insert whose table grows evicts for
// the growth's charge as well, since it is in usage by then, so usage stays
// within capacity insert by insert when entries shrink from 1000 bytes to
// 100 and the table grows to hold more of them.
TEST_F(FIFOCacheTest, NonStrictGrowthStaysWithinCapacity) {
  constexpr size_t kCapacity = 64 << 10;
  std::shared_ptr<Cache> cache =
      NewCache(kCapacity, /*strict*/ false, kFullChargeCacheMetadata);
  for (uint32_t i = 0; i < 200; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), 1000));
  }
  const size_t slots = cache->GetTableAddressCount();
  for (uint32_t i = 200; i < 600; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), 100));
    EXPECT_LE(cache->GetUsage(), kCapacity) << i;
  }
  EXPECT_GT(cache->GetTableAddressCount(), slots);
  EXPECT_TRUE(Contains(cache.get(), K(599)));
  EXPECT_TRUE(Holds(cache.get(), K(599)));
}

// An insert evicts for the slots the table grows by only when they are
// charged: under kDontChargeCacheMetadata an insert over capacity evicts
// its charge and HyperClockCache's extra (1 here), not the 144 bytes of
// its two new slots, which would take the whole overrun of 6.
TEST_F(FIFOCacheTest, UnchargedGrowthIsNotEvictedFor) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 0; i < 15; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
    pinned.push_back(h);
  }
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
  ASSERT_EQ(15, cache->GetUsage());
  ASSERT_EQ(24, cache->GetTableAddressCount());
  ASSERT_OK(Insert(cache.get(), K("y"), 1));
  EXPECT_EQ(26, cache->GetTableAddressCount());
  EXPECT_EQ(14, cache->GetUsage());
  EXPECT_EQ(14, Keys(cache.get()).size());
  EXPECT_TRUE(Holds(cache.get(), K("y")));
}

// A probation ring charged in a rotation during an insert's own eviction
// (the pinned head is in the first ring, the second is full and the table
// has outgrown it) is evicted for before the insert returns, by
// CreateStandalone as by Insert, so usage ends within capacity.
TEST_F(FIFOCacheTest, RingChargedDuringEvictionIsEvictedFor) {
  for (bool standalone : {true, false}) {
    SCOPED_TRACE(standalone);
    std::shared_ptr<Cache> cache =
        NewCache(1 << 20, /*strict*/ false, kFullChargeCacheMetadata);
    FIFOCache* fifo = Fifo(cache.get());
    Cache::Handle* pinned = nullptr;
    ASSERT_OK(Insert(cache.get(), K(0), 100, &pinned));
    // The first ring has 64 cells and the second 128.
    for (uint32_t i = 1; i < 64 + 128; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), 100));
    }
    cache->SetCapacity(cache->GetUsage());
    const size_t metadata = fifo->TEST_GetMetadataCharge();
    Cache::Handle* h = nullptr;
    if (standalone) {
      h = cache->CreateStandalone("s", nullptr, &kNoopCacheItemHelper, 100,
                                  /*allow_uncharged*/ false);
      ASSERT_NE(nullptr, h);
    } else {
      ASSERT_OK(Insert(cache.get(), K("new"), 100));
    }
    // A third ring, of 256 cells.
    EXPECT_LE(metadata + 256 * kQueueCellCharge,
              fifo->TEST_GetMetadataCharge());
    EXPECT_LE(cache->GetUsage(), cache->GetCapacity());
    if (h != nullptr) {
      EXPECT_TRUE(cache->Release(h));
    }
    EXPECT_FALSE(cache->Release(pinned));
  }
}

// The table never grows past its reservation (2048 slots here): at its
// occupancy limit (1229) unpinned entries are evicted by occupancy, and
// once every entry it can hold is pinned, a non-strict insert with a handle
// gets an entry outside the table, one without is dropped, and a strict
// one fails.
TEST_F(FIFOCacheTest, FullTableEvictsByOccupancyOrGoesStandalone) {
  std::shared_ptr<Cache> cache = NewCache(2048 * 64);
  for (uint32_t i = 0; i < 3 * 2048; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  EXPECT_EQ(2048, cache->GetTableAddressCount());
  EXPECT_EQ(1229, cache->GetOccupancyCount());
  cache->EraseUnRefEntries();
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 0; i < 1229; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(10000 + i), 1, &h));
    pins.push_back(h);
  }
  EXPECT_EQ(1229, cache->GetOccupancyCount());
  Cache::Handle* outside = nullptr;
  Status s = Insert(cache.get(), K("outside"), 1, &outside);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  ASSERT_NE(nullptr, outside);
  EXPECT_EQ(1229, cache->GetOccupancyCount());
  EXPECT_FALSE(Contains(cache.get(), K("outside")));
  EXPECT_EQ(1230, cache->GetPinnedUsage());
  EXPECT_TRUE(cache->Release(outside));
  ASSERT_OK(Insert(cache.get(), K("dropped")));
  EXPECT_FALSE(Holds(cache.get(), K("dropped")));
  EXPECT_EQ(1229, cache->GetUsage());
  cache->SetStrictCapacityLimit(true);
  Cache::Handle* strict = nullptr;
  s = Insert(cache.get(), K("strict"), 1, &strict);
  EXPECT_TRUE(s.IsMemoryLimit()) << s.ToString();
  EXPECT_EQ(nullptr, strict);
  for (Cache::Handle* h : pins) {
    EXPECT_FALSE(cache->Release(h));
  }
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// At the occupancy limit each insert of an entry of no charge (as
// CacheWithSecondaryAdapter inserts placeholders) evicts one entry for its
// slot, not every entry of no charge it meets, under either limit.
TEST_F(FIFOCacheTest, OccupancyEvictionFreesOneEntryPerInsert) {
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict);
    std::shared_ptr<Cache> cache = NewCache(2048 * 64, strict);
    std::vector<std::string> evicted;
    RecordEvictions(cache.get(), &evicted);
    size_t most_per_insert = 0;
    for (uint32_t i = 0; i < 3000; ++i) {
      const size_t before = evicted.size();
      ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 0));
      most_per_insert = std::max(most_per_insert, evicted.size() - before);
    }
    EXPECT_EQ(2048, cache->GetTableAddressCount());
    EXPECT_EQ(1, most_per_insert);
    EXPECT_EQ(3000, cache->GetOccupancyCount() + evicted.size());
  }
}

// At the occupancy limit an insert of no charge goes on past probation
// heads it promotes until it evicts an entry for its slot, under either
// limit: here K(0) to K(9) are promoted and K(10) is evicted.
TEST_F(FIFOCacheTest, OccupancyEvictionPromotesUntilItFrees) {
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict);
    std::shared_ptr<Cache> cache = NewCache(2048 * 64, strict);
    FIFOCache* fifo = Fifo(cache.get());
    std::vector<std::string> evicted;
    RecordEvictions(cache.get(), &evicted);
    for (uint32_t i = 0; i < 1229; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 0));
    }
    ASSERT_EQ(1229, cache->GetOccupancyCount());
    for (uint32_t i = 0; i < 10; ++i) {
      HitNTimes(cache.get(), K(i), 2);
    }
    ASSERT_OK(Insert(cache.get(), K("new"), /*charge*/ 0));
    EXPECT_TRUE(Holds(cache.get(), K("new")));
    EXPECT_EQ(std::vector<std::string>{K(10)}, evicted);
    EXPECT_EQ(1229, cache->GetOccupancyCount());
    for (uint32_t i = 0; i < 10; ++i) {
      EXPECT_FALSE(fifo->TEST_InProbation(K(i))) << i;
    }
  }
}

// A batch size near SIZE_MAX visits every entry in one call and finishes,
// on a shard and through the sharded ApplyToAllEntries.
TEST_F(FIFOCacheTest, ApplyToSomeEntriesHugeBatchFinishes) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ false, kDontChargeCacheMetadata,
               /*num_shard_bits*/ 1);
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  for (size_t batch : {SIZE_MAX, SIZE_MAX - 1}) {
    SCOPED_TRACE(batch);
    Cache::ApplyToAllEntriesOptions all;
    all.average_entries_per_lock = batch;
    size_t entries = 0;
    cache->ApplyToAllEntries([&](const Slice&, Cache::ObjectPtr, size_t,
                                 const Cache::CacheItemHelper*) { ++entries; },
                             all);
    EXPECT_EQ(10, entries);
  }
}

// Inserts, lookups, erases, releases that erase, SetCapacity and
// ApplyToAllEntries from four threads over four shards keep the accounting
// whole: afterwards nothing is pinned, usage is at least the charge of the
// visible entries (entries erased while another thread held a transient
// reference wait for eviction, as in HyperClockCache), and EraseUnRefEntries
// takes usage and occupancy to zero.
TEST_F(FIFOCacheTest, ConcurrentOpsKeepUsageConsistent) {
  std::shared_ptr<Cache> cache = NewCache(
      /*capacity*/ 1000 * kU, /*strict*/ false, kDontChargeCacheMetadata,
      /*num_shard_bits*/ 2);
  constexpr int kThreads = 4;
  std::vector<port::Thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&cache, t] {
      Random rnd(301 + t);
      std::vector<Cache::Handle*> held;
      for (int i = 0; i < 20000; ++i) {
        const std::string key = K(rnd.Uniform(400));
        switch (rnd.Uniform(6)) {
          case 0: {
            Cache::Handle* h = nullptr;
            Status s = Insert(cache.get(), key, (1 + rnd.Uniform(10)) * kU,
                              rnd.OneIn(2) ? &h : nullptr);
            EXPECT_TRUE(s.ok()) << s.ToString();
            if (h != nullptr) {
              held.push_back(h);
            }
            break;
          }
          case 1:
          case 2:
            if (Cache::Handle* h = cache->Lookup(key)) {
              held.push_back(h);
            }
            break;
          case 3:
            cache->Erase(key);
            break;
          case 4:
            if (rnd.OneIn(50)) {
              cache->SetCapacity((rnd.OneIn(2) ? 500 : 1000) * kU);
            }
            break;
          default:
            if (rnd.OneIn(50)) {
              cache->ApplyToAllEntries(
                  [](const Slice&, Cache::ObjectPtr, size_t,
                     const Cache::CacheItemHelper*) {},
                  {});
            }
            break;
        }
        if (held.size() > 8 || (!held.empty() && rnd.OneIn(3))) {
          const size_t idx = rnd.Uniform(static_cast<int>(held.size()));
          cache->Release(held[idx], /*erase_if_last_ref*/ rnd.OneIn(10));
          held[idx] = held.back();
          held.pop_back();
        }
      }
      for (Cache::Handle* h : held) {
        cache->Release(h);
      }
    });
  }
  for (port::Thread& thread : threads) {
    thread.join();
  }
  size_t charges = 0;
  cache->ApplyToAllEntries(
      [&](const Slice&, Cache::ObjectPtr, size_t charge,
          const Cache::CacheItemHelper*) { charges += charge; },
      {});
  EXPECT_EQ(0, cache->GetPinnedUsage());
  EXPECT_LE(charges, cache->GetUsage());
  cache->EraseUnRefEntries();
  EXPECT_EQ(0, cache->GetUsage());
  EXPECT_EQ(0, cache->GetOccupancyCount());
}

// Lookups, inserts, erases and releases race EraseUnRefEntries: every
// value a holder reads is alive and its own, every value is freed once,
// and once all pins are released and the table emptied, usage is the
// table's metadata alone (it never shrinks, so that is its grown size).
TEST_F(FIFOCacheTest, ConcurrentEraseUnRefEntriesKeepsLifetimeExact) {
  {
    std::shared_ptr<Cache> cache =
        NewCache(64 << 10, /*strict*/ false, kFullChargeCacheMetadata,
                 /*num_shard_bits*/ 1);
    constexpr int kThreads = 6;
    std::vector<port::Thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&cache, t] {
        Random rnd(300 + t);
        std::vector<std::pair<Cache::Handle*, uint32_t>> held;
        for (int i = 0; i < 20000; ++i) {
          const uint32_t k = rnd.Skewed(9);
          const uint32_t op = rnd.Uniform(100);
          if (op < 25) {
            Cache::Handle* h = nullptr;
            Status s = cache->Insert(L(k), new CountedValue(k), CountedHelper(),
                                     64 + rnd.Uniform(512),
                                     rnd.OneIn(2) ? &h : nullptr);
            ASSERT_TRUE(s.ok()) << s.ToString();
            if (h != nullptr) {
              CheckCountedValue(cache.get(), h, k);
              held.emplace_back(h, k);
            }
          } else if (op < 30) {
            cache->Erase(L(k));
          } else if (op < 32) {
            cache->EraseUnRefEntries();
          } else if (Cache::Handle* h = cache->Lookup(L(k))) {
            CheckCountedValue(cache.get(), h, k);
            held.emplace_back(h, k);
          }
          if (held.size() > 4 || (!held.empty() && rnd.OneIn(2))) {
            const size_t idx = rnd.Uniform(static_cast<int>(held.size()));
            CheckCountedValue(cache.get(), held[idx].first, held[idx].second);
            cache->Release(held[idx].first,
                           /*erase_if_last_ref*/ rnd.OneIn(10));
            held[idx] = held.back();
            held.pop_back();
          }
        }
        for (const auto& [h, k] : held) {
          CheckCountedValue(cache.get(), h, k);
          cache->Release(h);
        }
      });
    }
    for (port::Thread& thread : threads) {
      thread.join();
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
    cache->EraseUnRefEntries();
    EXPECT_EQ(0, cache->GetOccupancyCount());
    EXPECT_EQ(Fifo(cache.get())->TEST_GetMetadataCharge(), cache->GetUsage());
  }
  EXPECT_EQ(0, LiveValues().load());
}

// use_adaptive_mutex has nothing left to configure but still parses, true
// or false, in a fifo_cache:// URI and in FIFOCacheOptions, so
// configurations written for the mutex-based FIFOCache keep loading. An
// OPTIONS file carries the URI as the block_cache id, which holds a single
// option, so it loads the cache from a URI with capacity alone, and one
// with use_adaptive_mutex too fails to load.
TEST_F(FIFOCacheTest, UseAdaptiveMutexStillParses) {
  for (bool adaptive : {true, false}) {
    SCOPED_TRACE(adaptive);
    ConfigOptions config_options;
    std::shared_ptr<Cache> cache;
    ASSERT_OK(Cache::CreateFromString(
        config_options,
        std::string("fifo_cache://capacity=1M;use_adaptive_mutex=") +
            (adaptive ? "true" : "false"),
        &cache));
    ASSERT_NE(nullptr, dynamic_cast<FIFOCache*>(cache.get()));
    ASSERT_OK(Insert(cache.get(), K("a")));
    EXPECT_TRUE(Contains(cache.get(), K("a")));

    FIFOCacheOptions opts(1 << 20, /*num_shard_bits*/ 0,
                          /*strict_capacity_limit*/ false, nullptr, adaptive,
                          kDontChargeCacheMetadata);
    cache = opts.MakeSharedCache();
    ASSERT_NE(nullptr, cache);
    ASSERT_OK(Insert(cache.get(), K("a")));
    EXPECT_TRUE(Contains(cache.get(), K("a")));
  }

  Env* env = Env::Default();
  const std::string dir = test::PerThreadDBPath(env, "fifo_options_file");
  ASSERT_OK(env->CreateDirIfMissing(dir));
  const std::string path = dir + "/OPTIONS-000005";
  auto load = [&](const std::string& block_cache,
                  std::vector<ColumnFamilyDescriptor>* cf_descs) {
    EXPECT_OK(WriteStringToFile(env,
                                "[Version]\n"
                                "  rocksdb_version=11.8.1\n"
                                "  options_file_version=1.1\n"
                                "[DBOptions]\n"
                                "[CFOptions \"default\"]\n"
                                "[TableOptions/BlockBasedTable \"default\"]\n"
                                "  block_cache=" +
                                    block_cache + "\n",
                                path));
    ConfigOptions load_options;
    load_options.env = env;
    DBOptions db_options;
    return LoadOptionsFromFile(load_options, path, &db_options, cf_descs);
  };
  std::vector<ColumnFamilyDescriptor> cf_descs;
  ASSERT_OK(load("{id=fifo_cache://capacity=1M}", &cf_descs));
  ASSERT_EQ(1, cf_descs.size());
  const auto* table_options =
      cf_descs[0].options.table_factory->GetOptions<BlockBasedTableOptions>();
  ASSERT_NE(nullptr, table_options);
  ASSERT_NE(nullptr, table_options->block_cache);
  EXPECT_STREQ("FIFOCache", table_options->block_cache->Name());
  EXPECT_EQ(size_t{1} << 20, table_options->block_cache->GetCapacity());
  EXPECT_TRUE(
      load("{id=fifo_cache://capacity=1M;use_adaptive_mutex=true}", &cf_descs)
          .IsInvalidArgument());
  ASSERT_OK(DestroyDir(env, dir));
}

// A large cache reserves its table (capacity / 64 slots of 64 bytes, with
// their probation words, ghost tags and the probation queue's rings)
// without committing it: the table works, and on Linux each mapping is
// flagged no-reserve ("nr" in smaps), so it does not count against the
// commit limit. Linux ignores the flag, and would account the whole
// reservation, under vm.overcommit_memory=2.
TEST_F(FIFOCacheTest, LargeCapacityReservesWithoutCommit) {
#ifdef OS_WIN
  ROCKSDB_GTEST_BYPASS("Windows commits the reservation up front");
  return;
#endif  // OS_WIN
#ifdef OS_LINUX
  int overcommit_mode = 0;
  std::ifstream("/proc/sys/vm/overcommit_memory") >> overcommit_mode;
  if (overcommit_mode == 2) {
    ROCKSDB_GTEST_BYPASS("strict overcommit accounts the reservation");
    return;
  }
#endif  // OS_LINUX
  // One 16 GiB shard: 2^28 slots, a 16 GiB slot mapping.
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ size_t{16} << 30);
  ASSERT_OK(Insert(cache.get(), K(0)));
  EXPECT_TRUE(Contains(cache.get(), K(0)));
  const std::vector<const void*> mappings =
      Fifo(cache.get())->TEST_GetMappings();
  ASSERT_EQ(4, mappings.size());
#ifdef OS_LINUX
  for (const void* mapping : mappings) {
    ASSERT_NE(nullptr, mapping);
    const uintptr_t start = reinterpret_cast<uintptr_t>(mapping);
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool in_mapping = false;
    std::string flags;
    while (std::getline(smaps, line)) {
      unsigned long long lo = 0;
      unsigned long long hi = 0;
      char dash = 0;
      std::istringstream header(line);
      if (header >> std::hex >> lo >> dash >> hi && dash == '-') {
        in_mapping = lo <= start && start < hi;
      } else if (in_mapping && line.rfind("VmFlags:", 0) == 0) {
        flags = line + " ";
      }
    }
    EXPECT_NE(std::string::npos, flags.find(" nr ")) << flags;
  }
#endif  // OS_LINUX
}

// Lowering capacity evicts at once, oldest first, where HyperClockCache
// waits for the next insert; raising it evicts nothing. A pinned entry
// stays through a cut to 0, and once released, the next insert evicts it.
TEST_F(FIFOCacheTest, SetCapacityEvictsAtOnce) {
  std::shared_ptr<Cache> cache = NewCache(5 * kU);
  for (const char* key : {"a", "b", "c", "d", "e"}) {
    ASSERT_OK(Insert(cache.get(), K(key), kU));
  }
  cache->SetCapacity(2 * kU);
  EXPECT_EQ(2 * kU, cache->GetUsage());
  for (const char* key : {"a", "b", "c"}) {
    EXPECT_FALSE(Holds(cache.get(), K(key))) << key;
  }
  for (const char* key : {"d", "e"}) {
    EXPECT_TRUE(Holds(cache.get(), K(key))) << key;
  }
  cache->SetCapacity(10 * kU);
  EXPECT_EQ(2 * kU, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K("d")));

  Cache::Handle* h = cache->Lookup(K("d"));
  ASSERT_NE(nullptr, h);
  cache->SetCapacity(0);
  EXPECT_EQ(kU, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K("d")));
  EXPECT_FALSE(cache->Release(h));
  ASSERT_OK(Insert(cache.get(), K("f"), kU));
  EXPECT_FALSE(Holds(cache.get(), K("d")));
}

// A cut evicts everything it needs that isn't pinned, though each eviction
// call gives up after 30 pins with nothing freed: here 40 pinned entries
// are at the head of probation. With everything pinned it stops.
TEST_F(FIFOCacheTest, SetCapacityEvictsPastManyPins) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 0; i < 100; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), kU, i < 40 ? &h : nullptr));
    if (h != nullptr) {
      pinned.push_back(h);
    }
  }
  cache->SetCapacity(50 * kU);
  EXPECT_EQ(50 * kU, cache->GetUsage());
  for (uint32_t i = 0; i < 100; ++i) {
    EXPECT_EQ(i < 40 || i >= 90, Holds(cache.get(), K(i))) << i;
  }
  cache->SetCapacity(0);
  EXPECT_EQ(40 * kU, cache->GetUsage());
  EXPECT_EQ(40, cache->GetOccupancyCount());
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// An eviction call gives up once it has seen 30 pins per entry freed, plus
// 30, so with 40 pinned entries after each unpinned one the first call
// frees U0 to U2 and stops; a cut that needs five keeps calling and frees
// U3 and U4 too.
TEST_F(FIFOCacheTest, SetCapacityEvictsPastPinsBetweenFrees) {
  std::shared_ptr<Cache> cache = NewCache(205 * kU);
  std::vector<Cache::Handle*> pinned;
  for (uint32_t g = 0; g < 5; ++g) {
    ASSERT_OK(Insert(cache.get(), "U" + std::to_string(g), kU));
    for (uint32_t i = 0; i < 40; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(g * 40 + i), kU, &h));
      pinned.push_back(h);
    }
  }
  ASSERT_EQ(205 * kU, cache->GetUsage());
  cache->SetCapacity(200 * kU);
  EXPECT_EQ(200 * kU, cache->GetUsage());
  for (uint32_t g = 0; g < 5; ++g) {
    EXPECT_FALSE(Holds(cache.get(), "U" + std::to_string(g))) << g;
  }
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// Probation's share follows capacity, so a cut that takes probation over
// its new share evicts from probation first, even when it was under the
// share capacity gave before: here the oldest probation entries go and
// the promoted entry stays.
TEST_F(FIFOCacheTest, SetCapacityCutDrawsProbationFirst) {
  std::shared_ptr<Cache> cache = NewCache(50 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("r"), kU));
  HitNTimes(cache.get(), K("r"), 2);
  for (uint32_t i = 0; i < 49; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_EQ(50 * kU, cache->GetUsage());
  // Promotes r and evicts K(0).
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K("r")));
  ASSERT_FALSE(Holds(cache.get(), K(0)));
  ASSERT_EQ(49 * kU, fifo->TEST_GetProbationUsage());
  cache->SetCapacity(10 * kU);
  EXPECT_EQ(10 * kU, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K("r")));
  EXPECT_EQ(9 * kU, fifo->TEST_GetProbationUsage());
  for (uint32_t i = 1; i < 49; ++i) {
    EXPECT_EQ(i >= 41, Holds(cache.get(), K(i))) << i;
  }

  // Probation at 10 is under the share of 12 at 100 units, and over the
  // share of 6 at 50, so the cut evicts K(101) first.
  cache = NewCache(100 * kU);
  fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 90 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  for (uint32_t i = 100; i < 110; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // Promotes big and evicts K(100).
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K("big")));
  ASSERT_EQ(10 * kU, fifo->TEST_GetProbationUsage());
  cache->SetCapacity(50 * kU);
  EXPECT_FALSE(Holds(cache.get(), K(101)));
  EXPECT_LE(cache->GetUsage(), 50 * kU);
}

// A cut counts the probation entries it finds pinned out of probation's
// share as it goes: with most of probation pinned, the resident entry pays
// for the cut and the unpinned probation entries stay.
TEST_F(FIFOCacheTest, SetCapacityCountsProbationPinsItFinds) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 80 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  for (uint32_t i = 0; i < 20; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // Promotes big and evicts K(0): probation holds K(1) to K(19) and t, 20
  // units against a share of 12.
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K("big")));
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 1; i <= 15; ++i) {
    pinned.push_back(cache->Lookup(K(i)));
    ASSERT_NE(nullptr, pinned.back());
  }
  // Unpinned probation is 5 against a share of 10.8 once the cut has seen
  // the pins, so big goes. Counting the pins as unpinned would evict K(16)
  // to K(19) and t first.
  cache->SetCapacity(90 * kU);
  EXPECT_FALSE(Holds(cache.get(), K("big")));
  EXPECT_TRUE(Holds(cache.get(), K(16)));
  EXPECT_TRUE(Holds(cache.get(), K("t")));
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// An entry larger than probation's whole share waits in probation like any
// other and is evicted or promoted on its turn.
TEST_F(FIFOCacheTest, SingleEntryExceedsProbationBudget) {
  {
    std::shared_ptr<Cache> cache = NewCache(100 * kU);
    FIFOCache* fifo = Fifo(cache.get());
    ASSERT_OK(Insert(cache.get(), K("big"), 20 * kU));
    EXPECT_EQ(20 * kU, fifo->TEST_GetProbationUsage());
    for (uint32_t i = 0; i < 80; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), kU));
    }
    EXPECT_EQ(100 * kU, cache->GetUsage());
    ASSERT_OK(Insert(cache.get(), K("t"), kU));
    EXPECT_FALSE(Holds(cache.get(), K("big")));
    EXPECT_TRUE(Holds(cache.get(), K("t")));
    EXPECT_EQ(81 * kU, fifo->TEST_GetProbationUsage());
    EXPECT_EQ(81 * kU, cache->GetUsage());
  }
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 20 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  for (uint32_t i = 0; i < 80; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  EXPECT_TRUE(Holds(cache.get(), K("big")));
  EXPECT_FALSE(Holds(cache.get(), K(0)));
  EXPECT_EQ(80 * kU, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(100 * kU, cache->GetUsage());
}

// With probation over its share and every entry pinned, eviction gives up
// and the insert goes over capacity. With an entry the resident clock can
// evict, the pins leave probation's share as eviction steps see them, at
// most 30 a call (the eviction effort cap: 30 pinned entries seen for each
// one freed), and once all 99 have, which takes four inserts, the resident
// entry goes.
TEST_F(FIFOCacheTest, FullyPinnedProbationOverBudgetTerminates) {
  {
    std::shared_ptr<Cache> cache = NewCache(100 * kU);
    FIFOCache* fifo = Fifo(cache.get());
    std::vector<Cache::Handle*> pinned;
    for (uint32_t i = 0; i < 100; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), kU));
    }
    EXPECT_EQ(100 * kU, fifo->TEST_GetProbationUsage());
    for (uint32_t i = 0; i < 100; ++i) {
      pinned.push_back(cache->Lookup(K(i)));
      ASSERT_NE(nullptr, pinned.back());
    }
    ASSERT_OK(Insert(cache.get(), K("x"), kU));
    EXPECT_TRUE(Holds(cache.get(), K("x")));
    EXPECT_EQ(101 * kU, cache->GetUsage());
    for (Cache::Handle* h : pinned) {
      EXPECT_FALSE(cache->Release(h));
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
  }
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("r"), kU));
  HitNTimes(cache.get(), K("r"), 2);
  for (uint32_t i = 0; i < 99; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // Promotes r, evicts K(0).
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_TRUE(Holds(cache.get(), K("r")));
  ASSERT_FALSE(Holds(cache.get(), K(0)));
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 1; i < 99; ++i) {
    pinned.push_back(cache->Lookup(K(i)));
    ASSERT_NE(nullptr, pinned.back());
  }
  pinned.push_back(cache->Lookup(K("t")));
  ASSERT_NE(nullptr, pinned.back());
  EXPECT_EQ(99 * kU, fifo->TEST_GetProbationUsage());
  int inserts = 0;
  while (Holds(cache.get(), K("r")) && inserts < 10) {
    ASSERT_OK(Insert(cache.get(), K(200 + inserts), kU));
    ++inserts;
  }
  EXPECT_EQ(4, inserts);
  EXPECT_FALSE(Holds(cache.get(), K("r")));
  for (uint32_t i = 1; i < 99; ++i) {
    EXPECT_TRUE(Holds(cache.get(), K(i))) << i;
  }
  for (Cache::Handle* h : pinned) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// When the resident clock frees nothing (its only entry is pinned) and
// probation, within its share, then only promotes, the clock runs again and
// evicts the entry just promoted, which has no hits there.
TEST_F(FIFOCacheTest, ProbationFallbackPromotionRetriesResident) {
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict ? "strict" : "non-strict");
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 100, strict);
    FIFOCache* fifo = Fifo(cache.get());
    ASSERT_OK(Insert(cache.get(), K("r"), 89));
    HitNTimes(cache.get(), K("r"), 2);
    ASSERT_OK(Insert(cache.get(), K("s"), 11));
    // Promotes r, evicts s.
    ASSERT_OK(Insert(cache.get(), K("t")));
    cache->Erase(K("t"));
    Cache::Handle* hr = cache->Lookup(K("r"));
    ASSERT_NE(nullptr, hr);
    ASSERT_OK(Insert(cache.get(), K("p"), 10));
    HitNTimes(cache.get(), K("p"), 2);
    ASSERT_EQ(10, fifo->TEST_GetProbationUsage());
    ASSERT_EQ(99, cache->GetUsage());
    Cache::Handle* hx = nullptr;
    ASSERT_OK(Insert(cache.get(), K("x"), 2, &hx));
    ASSERT_NE(nullptr, hx);
    EXPECT_FALSE(Holds(cache.get(), K("p")));
    EXPECT_LE(cache->GetUsage(), 100);
    EXPECT_FALSE(cache->Release(hx));
    EXPECT_FALSE(cache->Release(hr));
  }
}

// The same when probation's fallback promotes one entry and evicts another
// that frees too little: the clock runs again and takes the promoted one.
TEST_F(FIFOCacheTest, ResidentRetriedAfterProbationPromotes) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 86, /*strict*/ true);
  ASSERT_OK(Insert(cache.get(), K("r"), 80));
  HitNTimes(cache.get(), K("r"), 2);
  ASSERT_OK(Insert(cache.get(), K("s"), 6));
  // Promotes r, evicts s.
  ASSERT_OK(Insert(cache.get(), K("t"), 1));
  cache->Erase(K("t"));
  Cache::Handle* hr = cache->Lookup(K("r"));
  ASSERT_NE(nullptr, hr);
  ASSERT_OK(Insert(cache.get(), K("p"), 5));
  HitNTimes(cache.get(), K("p"), 2);
  ASSERT_OK(Insert(cache.get(), K("c"), 1));
  // Probation's 6 are within its share of 10, so the clock goes first and
  // frees nothing (r is pinned). Probation promotes p and evicts c, freeing
  // 1 of the 2 needed; the clock then evicts p.
  Cache::Handle* hx = nullptr;
  ASSERT_OK(Insert(cache.get(), K("x"), 2, &hx));
  ASSERT_NE(nullptr, hx);
  EXPECT_FALSE(Holds(cache.get(), K("p")));
  EXPECT_FALSE(Holds(cache.get(), K("c")));
  EXPECT_LE(cache->GetUsage(), 86);
  EXPECT_FALSE(cache->Release(hx));
  EXPECT_FALSE(cache->Release(hr));
}

// With every entry pinned, releasing one, in either queue, makes it
// evictable again.
TEST_F(FIFOCacheTest, ReleasedEntryEvictableInFullyPinnedShard) {
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
    std::vector<Cache::Handle*> pinned;
    for (uint32_t i = 0; i < 10; ++i) {
      Cache::Handle* h = nullptr;
      ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
      pinned.push_back(h);
    }
    // Over capacity, as nothing can be evicted.
    ASSERT_OK(Insert(cache.get(), K("x")));
    EXPECT_EQ(11, cache->GetUsage());
    EXPECT_FALSE(cache->Release(pinned[3]));
    // Back to capacity: x and K(3) go.
    ASSERT_OK(Insert(cache.get(), K("y")));
    EXPECT_TRUE(Holds(cache.get(), K("y")));
    EXPECT_FALSE(Holds(cache.get(), K(3)));
    EXPECT_FALSE(Holds(cache.get(), K("x")));
    EXPECT_EQ(10, cache->GetUsage());
    for (uint32_t i = 0; i < 10; ++i) {
      if (i != 3) {
        cache->Release(pinned[i]);
      }
    }
  }
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  for (uint32_t i = 0; i < 9; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
    HitNTimes(cache.get(), K(i), 2);
  }
  ASSERT_OK(Insert(cache.get(), K("s")));
  // Promotes K(0)..K(8), evicts s.
  ASSERT_OK(Insert(cache.get(), K("t")));
  cache->Erase(K("t"));
  ASSERT_EQ(0, Fifo(cache.get())->TEST_GetProbationUsage());
  std::vector<Cache::Handle*> pinned;
  for (uint32_t i = 0; i < 9; ++i) {
    pinned.push_back(cache->Lookup(K(i)));
    ASSERT_NE(nullptr, pinned.back());
  }
  EXPECT_FALSE(cache->Release(pinned[4]));
  ASSERT_OK(Insert(cache.get(), K("y"), 2));
  EXPECT_TRUE(Holds(cache.get(), K("y")));
  EXPECT_FALSE(Holds(cache.get(), K(4)));
  EXPECT_EQ(10, cache->GetUsage());
  for (uint32_t i = 0; i < 9; ++i) {
    if (i != 4) {
      cache->Release(pinned[i]);
    }
  }
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// An entry larger than probation's whole share takes its room from
// probation first and from the resident queue only for the rest, then waits
// in probation over its share and goes on its turn. (Without the incoming
// charge in the queue choice, probation at its share would leave all 20 to
// the resident queue.)
TEST_F(FIFOCacheTest, OversizedEntryPaidByProbationFirst) {
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict ? "strict" : "non-strict");
    std::vector<std::string> evicted;
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 100, strict);
    RecordEvictions(cache.get(), &evicted);
    FIFOCache* fifo = Fifo(cache.get());
    std::set<std::string> resident;
    for (uint32_t i = 0; i < 8; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i), 11));
      HitNTimes(cache.get(), K(i), 2);
      resident.insert(K(i));
    }
    std::vector<std::string> probation;
    for (uint32_t i = 100; i < 112; ++i) {
      ASSERT_OK(Insert(cache.get(), K(i)));
      probation.push_back(K(i));
    }
    // Promotes K(0)..K(7), evicts K(100).
    ASSERT_OK(Insert(cache.get(), K("t")));
    ASSERT_EQ(std::vector<std::string>{K(100)}, evicted);
    ASSERT_EQ(12, fifo->TEST_GetProbationUsage());
    probation.push_back(K("t"));
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K("x"), 20, &h));
    ASSERT_NE(nullptr, h);
    EXPECT_FALSE(cache->Release(h));
    // Probation's 12 went first, in order, and the clock covered the rest.
    ASSERT_GE(evicted.size(), 14);
    EXPECT_EQ(probation,
              std::vector<std::string>(evicted.begin(), evicted.begin() + 13));
    for (size_t i = 13; i < evicted.size(); ++i) {
      EXPECT_EQ(1, resident.erase(evicted[i])) << i;
    }
    EXPECT_EQ(20, fifo->TEST_GetProbationUsage());
    EXPECT_EQ(20 + 11 * resident.size(), cache->GetUsage());
    // Over its share with a cold head, probation pays for the next insert
    // with x.
    const size_t before = evicted.size();
    ASSERT_OK(Insert(cache.get(), K("y"), 101 - cache->GetUsage()));
    EXPECT_EQ(
        (std::vector<std::string>{K("x")}),
        std::vector<std::string>(evicted.begin() + before, evicted.end()));
    for (const std::string& key : resident) {
      EXPECT_TRUE(Holds(cache.get(), key));
    }
  }
}

// On a cache whose probation share rounds to zero, every eviction starts in
// probation, which still sorts entries hit twice from those hit once.
TEST_F(FIFOCacheTest, ZeroBudgetMixedChargesKeepPromotion) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 8);
  FIFOCache* fifo = Fifo(cache.get());
  EXPECT_EQ(0, fifo->TEST_GetProbationTarget());
  ASSERT_OK(Insert(cache.get(), K("a"), 1));
  HitNTimes(cache.get(), K("a"), 2);
  ASSERT_OK(Insert(cache.get(), K("b"), 3));
  HitNTimes(cache.get(), K("b"), 1);
  ASSERT_OK(Insert(cache.get(), K("c"), 4));
  // Promotes a, evicts b.
  ASSERT_OK(Insert(cache.get(), K("d"), 2));
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_FALSE(Holds(cache.get(), K("b")));
  EXPECT_EQ(7, cache->GetUsage());
  EXPECT_EQ(6, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K("e"), 1));
  // Needs 5: c and d go from probation, and resident a is left alone.
  ASSERT_OK(Insert(cache.get(), K("f"), 5));
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_FALSE(Holds(cache.get(), K("c")));
  EXPECT_FALSE(Holds(cache.get(), K("d")));
  EXPECT_TRUE(Holds(cache.get(), K("e")));
  EXPECT_TRUE(Holds(cache.get(), K("f")));
  EXPECT_EQ(7, cache->GetUsage());
}

// Under a workload whose charges span 4 to 20000, more than three orders of
// magnitude, the largest over probation's share of 12000, usage never
// exceeds capacity after any operation and equals the sum of the cached
// charges, and every insert succeeds, strict inserts with a handle
// included, since nothing stays pinned.
TEST_F(FIFOCacheTest, MixedChargeWorkloadRespectsCapacity) {
  constexpr size_t kCapacity = 100000;
  const size_t kCharges[] = {4, 40, 400, 4000, 20000};
  for (bool strict : {false, true}) {
    SCOPED_TRACE(strict ? "strict" : "non-strict");
    std::shared_ptr<Cache> cache = NewCache(kCapacity, strict);
    Random rnd(301);
    for (int op = 0; op < 20000; ++op) {
      const uint32_t k = rnd.Uniform(2000);
      const std::string key = K(k);
      const size_t charge = kCharges[k % 5];
      const uint32_t action = rnd.Uniform(10);
      if (action < 7) {
        if (!Contains(cache.get(), key)) {
          Cache::Handle* h = nullptr;
          ASSERT_OK(Insert(cache.get(), key, charge, strict ? &h : nullptr));
          if (strict) {
            ASSERT_NE(nullptr, h);
            EXPECT_FALSE(cache->Release(h));
          }
        }
      } else if (action < 9) {
        ASSERT_OK(Insert(cache.get(), key, charge));
      } else {
        cache->Erase(key);
      }
      ASSERT_LE(cache->GetUsage(), kCapacity) << op;
      if (op % 1000 == 0) {
        size_t sum = 0;
        cache->ApplyToAllEntries(
            [&](const Slice& /*key*/, Cache::ObjectPtr /*value*/, size_t c,
                const Cache::CacheItemHelper* /*helper*/) { sum += c; },
            {});
        ASSERT_EQ(sum, cache->GetUsage()) << op;
      }
    }
    EXPECT_EQ(0, cache->GetPinnedUsage());
  }
}

// Tiny (4) and huge (20000, over probation's share of 12000) cold entries
// alternate around one hot entry of each size: usage stays within capacity,
// neither size starves, both hot entries hit on every round after the
// first, and every cold entry of either size is admitted.
TEST_F(FIFOCacheTest, AlternatingTinyAndHugeNoStarvation) {
  constexpr size_t kCapacity = 100000;
  std::shared_ptr<Cache> cache = NewCache(kCapacity);
  int tiny_hits = 0;
  int huge_hits = 0;
  constexpr int kRounds = 1000;
  for (int round = 0; round < kRounds; ++round) {
    if (Contains(cache.get(), K("hot_tiny"))) {
      ++tiny_hits;
    } else {
      ASSERT_OK(Insert(cache.get(), K("hot_tiny"), 4));
    }
    ASSERT_LE(cache->GetUsage(), kCapacity);
    if (Contains(cache.get(), K("hot_huge"))) {
      ++huge_hits;
    } else {
      ASSERT_OK(Insert(cache.get(), K("hot_huge"), 20000));
    }
    ASSERT_LE(cache->GetUsage(), kCapacity);
    const std::string tiny = K(2 * round);
    ASSERT_OK(Insert(cache.get(), tiny, 4));
    ASSERT_LE(cache->GetUsage(), kCapacity);
    ASSERT_TRUE(Holds(cache.get(), tiny));
    const std::string huge = K(2 * round + 1);
    ASSERT_OK(Insert(cache.get(), huge, 20000));
    ASSERT_LE(cache->GetUsage(), kCapacity);
    ASSERT_TRUE(Holds(cache.get(), huge));
  }
  EXPECT_EQ(kRounds - 1, tiny_hits);
  EXPECT_EQ(kRounds - 1, huge_hits);
}

// Erase, the last release of an erased entry and EraseUnRefEntries each
// take entries out of probation's usage, and eviction passes over their
// cells.
TEST_F(FIFOCacheTest, ErasedEntriesLeaveProbation) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a"), 3));
  Cache::Handle* hb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("b"), 2, &hb));
  ASSERT_OK(Insert(cache.get(), K("c")));
  ASSERT_OK(Insert(cache.get(), K("d")));
  EXPECT_EQ(7, fifo->TEST_GetProbationUsage());
  cache->Erase(K("a"));
  EXPECT_EQ(4, fifo->TEST_GetProbationUsage());
  // Erased while pinned: charged, and in probation, until its last release
  // frees it.
  cache->Erase(K("b"));
  EXPECT_EQ(4, fifo->TEST_GetProbationUsage());
  EXPECT_TRUE(cache->Release(hb));
  EXPECT_EQ(2, fifo->TEST_GetProbationUsage());
  // A pinned entry outlasts EraseUnRefEntries, in probation.
  Cache::Handle* hd = cache->Lookup(K("d"));
  ASSERT_NE(nullptr, hd);
  cache->EraseUnRefEntries();
  EXPECT_EQ(1, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_FALSE(cache->Release(hd));
  // The cells of a, b and c are stale, so d, the one live entry from
  // before, is the first evicted.
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  EXPECT_EQ(std::vector<std::string>{K("d")}, evicted);
  EXPECT_EQ(10, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(10, cache->GetUsage());
}

// Under concurrent inserts, hits, pins, releases and erases that keep
// entries moving through both queues, probation's usage stays within the
// cache's, its pinned part within it, and both come back to zero once
// everything is released and erased. (At destruction a Debug build also
// checks the counters against the slots.)
TEST_F(FIFOCacheTest, ConcurrentProbationAccountingStaysExact) {
  std::shared_ptr<Cache> cache = NewCache(64 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  constexpr int kThreads = 6;
  constexpr uint32_t kKeys = 300;
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      Random rnd(1000 + t);
      std::vector<Cache::Handle*> held;
      for (int op = 0; op < 20000; ++op) {
        // A hot fifth of the keys gets most hits, so promotions happen.
        const uint32_t k =
            rnd.OneIn(2) ? rnd.Uniform(kKeys / 5) : rnd.Uniform(kKeys);
        const size_t charge = kU * (1 + k % 3);
        switch (rnd.Uniform(8)) {
          case 0:
          case 1:
          case 2: {
            Cache::Handle* h = cache->Lookup(K(k));
            if (h == nullptr) {
              Cache::Handle** want =
                  held.size() < 8 && rnd.OneIn(4) ? &h : nullptr;
              ASSERT_OK(Insert(cache.get(), K(k), charge, want));
            }
            if (h != nullptr) {
              if (held.size() < 8 && rnd.OneIn(4)) {
                held.push_back(h);
              } else {
                cache->Release(h);
              }
            }
            break;
          }
          case 3:
            ASSERT_OK(Insert(cache.get(), K(k), charge));
            break;
          case 4:
            cache->Erase(K(k));
            break;
          case 5:
            if (!held.empty()) {
              cache->Release(held.back(),
                             /*erase_if_last_ref*/ rnd.OneIn(3));
              held.pop_back();
            }
            break;
          default:
            if (Cache::Handle* h = cache->Lookup(K(rnd.Uniform(kKeys / 5)))) {
              cache->Release(h);
            }
            break;
        }
      }
      for (Cache::Handle* h : held) {
        cache->Release(h);
      }
    });
  }
  for (port::Thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(0, cache->GetPinnedUsage());
  EXPECT_LE(fifo->TEST_GetProbationUsage(), cache->GetUsage());
  EXPECT_LE(fifo->TEST_GetProbationPinnedUsage(),
            fifo->TEST_GetProbationUsage());
  cache->EraseUnRefEntries();
  EXPECT_EQ(0, cache->GetUsage());
  EXPECT_EQ(0, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(0, fifo->TEST_GetProbationPinnedUsage());
}

// An insert into a full probation queue drops at most
// kMaxStaleDropsPerPush stale cells to make room. Here the head is in older
// rings, all stale, which free no room in the full newest ring, so the
// first insert gives up after 8 of their 12 and goes to the resident queue,
// and the next drops the other 4 and one in the newest ring, and is queued.
TEST_F(FIFOCacheTest, StaleCellDropsPerInsertAreBounded) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 1000);
  FIFOCache* fifo = Fifo(cache.get());
  // Rings of 4, 8 and 16 cells: the table starts with 4 slots and grows to
  // hold 13 entries, adding a ring each time the newest is full.
  for (uint32_t i = 0; i < 13; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  for (uint32_t i = 0; i < 13; ++i) {
    cache->Erase(K(i));
  }
  // The newest ring's other 15 cells, stale too.
  for (uint32_t i = 100; i < 115; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
    cache->Erase(K(i));
  }
  ASSERT_EQ(0, cache->GetUsage());
  ASSERT_EQ(0, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K("g")));
  EXPECT_EQ(0, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K("h")));
  EXPECT_EQ(1, fifo->TEST_GetProbationUsage());
  EXPECT_TRUE(Holds(cache.get(), K("g")));
  EXPECT_TRUE(Holds(cache.get(), K("h")));
}

// The limit on probation steps per eviction is a lap of the queue, which
// can hold more cells than the table has slots: here 51 stale cells (of
// entries erased while queued) lie ahead of q, more than the table's 47 or
// so slots, and probation still gets to q rather than handing the eviction
// to the resident clock.
TEST_F(FIFOCacheTest, ProbationStepLimitCoversTheWholeQueue) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  // Rings of 4, 8, 16 and 32 cells, for a table grown to hold 29 entries
  // (of no charge, so nothing is evicted).
  for (uint32_t i = 0; i < 29; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 0));
  }
  for (uint32_t i = 0; i < 29; ++i) {
    cache->Erase(K(i));
  }
  ASSERT_LT(cache->GetTableAddressCount(), 51);
  for (uint32_t i = 100; i < 130; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 0));
    cache->Erase(K(i));
  }
  // q fills the newest ring. r finds it full, drops 8 stale cells, and goes
  // to the resident queue.
  ASSERT_OK(Insert(cache.get(), K("q")));
  ASSERT_OK(Insert(cache.get(), K("r"), /*charge*/ 9));
  ASSERT_EQ(1, fifo->TEST_GetProbationUsage());
  ASSERT_EQ(10, cache->GetUsage());
  ASSERT_OK(Insert(cache.get(), K("x")));
  EXPECT_EQ(std::vector<std::string>{K("q")}, evicted);
  EXPECT_TRUE(Holds(cache.get(), K("r")));
}

// A stale cell is skipped whatever its slot holds now. Here every key was
// erased and inserted again with a handle, landing in its old slot, so each
// slot's new entry is pinned: counting the old cells as pinned heads too
// would take the eviction to HyperClockCache's effort cap (30 pins seen
// with nothing freed) before it reached u.
TEST_F(FIFOCacheTest, StaleCellsOfRefilledSlotsDoNotCountAsPins) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 26);
  RecordEvictions(cache.get(), &evicted);
  for (uint32_t i = 0; i < 25; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 0; i < 25; ++i) {
    cache->Erase(K(i));
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), 1, &h));
    pins.push_back(h);
  }
  ASSERT_OK(Insert(cache.get(), K("u")));
  ASSERT_EQ(26, cache->GetUsage());
  ASSERT_OK(Insert(cache.get(), K("x")));
  EXPECT_EQ(std::vector<std::string>{K("u")}, evicted);
  EXPECT_EQ(26, cache->GetUsage());
  for (Cache::Handle* h : pins) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// A probation victim inserted again goes to the resident queue, so the
// pressure that evicted it once no longer does: with only a resident,
// probation stays over its share and pays for every insert.
TEST_F(FIFOCacheTest, GhostReadmitsProbationVictimToResident) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  for (uint32_t i = 0; i < 99; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_EQ(std::vector<std::string>{K("a")}, evicted);
  EXPECT_TRUE(fifo->TEST_InGhost(K("a")));
  // Evicts K(0); a goes to the resident queue, taking its ghost.
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
  EXPECT_FALSE(fifo->TEST_InGhost(K("a")));
  // A full probation's worth of cold inserts, which would take a there.
  for (uint32_t i = 200; i < 300; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_EQ(100 * kU, cache->GetUsage());
}

// A resident victim leaves no ghost and comes back to probation, while a
// probation victim from the same run comes back to the resident queue.
TEST_F(FIFOCacheTest, GhostIgnoresResidentVictims) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 88 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  HitNTimes(cache.get(), K("a"), 2);
  for (uint32_t i = 0; i < 11; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_EQ(100 * kU, cache->GetUsage());
  // Promotes big and a, and evicts K(0) from probation.
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_FALSE(Holds(cache.get(), K(0)));
  ASSERT_EQ(11 * kU, fifo->TEST_GetProbationUsage());
  HitNTimes(cache.get(), K("big"), 1);
  // u lands probation at its share, so the resident clock pays: big spends
  // its hit and a goes.
  ASSERT_OK(Insert(cache.get(), K("u"), kU));
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_TRUE(Holds(cache.get(), K("big")));
  EXPECT_FALSE(fifo->TEST_InGhost(K("a")));
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_TRUE(fifo->TEST_InProbation(K("a")));
  ASSERT_OK(Insert(cache.get(), K(0), kU));
  EXPECT_TRUE(Holds(cache.get(), K(0)));
  EXPECT_FALSE(fifo->TEST_InProbation(K(0)));
}

// Readmission takes the ghost, and an entry readmitted at no hits doesn't
// count toward probation's share. After a is readmitted and then evicted
// from the resident queue, inserting it again puts it in probation.
TEST_F(FIFOCacheTest, GhostForgetsReadmittedKey) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("big"), 87 * kU));
  HitNTimes(cache.get(), K("big"), 2);
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  for (uint32_t i = 0; i < 12; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_EQ(100 * kU, cache->GetUsage());
  // Promotes big, evicts a.
  ASSERT_OK(Insert(cache.get(), K("t"), kU));
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  // Evicts K(0); a is readmitted.
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
  EXPECT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  HitNTimes(cache.get(), K("big"), 1);
  // K(0) is readmitted, so probation stays at its share and the resident
  // clock pays: big spends its hit and a, at no hits, goes.
  ASSERT_OK(Insert(cache.get(), K(0), kU));
  EXPECT_FALSE(fifo->TEST_InProbation(K(0)));
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_TRUE(Holds(cache.get(), K("big")));
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_TRUE(fifo->TEST_InProbation(K("a")));
}

// After many victims, the newest is remembered.
TEST_F(FIFOCacheTest, GhostRemembersNewestVictim) {
  std::shared_ptr<Cache> cache = NewCache(40 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  for (uint32_t i = 0; i < 40; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // Victims K(0)..K(39), then K(100)..K(159).
  for (uint32_t i = 100; i < 200; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_FALSE(Holds(cache.get(), K(159)));
  EXPECT_TRUE(fifo->TEST_InGhost(K(159)));
  ASSERT_OK(Insert(cache.get(), K(159), kU));
  EXPECT_FALSE(fifo->TEST_InProbation(K(159)));
}

// The ghost keeps 32 bits of a key's hash at its home, so another key with
// the same home and the same 32 bits is readmitted as if it were the victim.
TEST_F(FIFOCacheTest, GhostTagCollisionAdmitsToResident) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  UniqueId64x2 hashed = Shard::ComputeHash(K("a"), cache->GetHashSeed());
  hashed[0] ^= uint64_t{1} << 63;
  UniqueId64x2 unhashed;
  const std::string collider =
      Shard::ReverseHash(hashed, &unhashed, cache->GetHashSeed()).ToString();
  ASSERT_NE(K("a"), collider);
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_TRUE(fifo->TEST_InGhost(collider));
  ASSERT_OK(Insert(cache.get(), collider, kU));
  EXPECT_FALSE(fifo->TEST_InProbation(collider));
  EXPECT_FALSE(fifo->TEST_InGhost(K("a")));
}

// An insert takes its key's tag before it evicts, so the victim it evicts
// can't replace the tag first, even when the victim's key has the same home.
TEST_F(FIFOCacheTest, GhostTagTakenBeforeTheInsertEvicts) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  // The same home, from the same hashed_key[1], and another tag.
  UniqueId64x2 hashed = Shard::ComputeHash(K("a"), cache->GetHashSeed());
  hashed[0] ^= 0x10;
  UniqueId64x2 unhashed;
  const std::string c =
      Shard::ReverseHash(hashed, &unhashed, cache->GetHashSeed()).ToString();
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  ASSERT_OK(Insert(cache.get(), c, kU));
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  ASSERT_TRUE(fifo->TEST_InGhost(K("a")));
  // Evicts c, whose tag goes where a's was.
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_FALSE(Holds(cache.get(), c));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
  EXPECT_TRUE(fifo->TEST_InGhost(c));
}

// A strict insert that can't make room puts back the tag it took, unless a
// victim it evicted before giving up has replaced it.
TEST_F(FIFOCacheTest, GhostTagPutBackWhenStrictInsertFails) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU, /*strict*/ true);
  FIFOCache* fifo = Fifo(cache.get());
  UniqueId64x2 hashed = Shard::ComputeHash(K("a"), cache->GetHashSeed());
  hashed[0] ^= 0x10;
  UniqueId64x2 unhashed;
  const std::string c =
      Shard::ReverseHash(hashed, &unhashed, cache->GetHashSeed()).ToString();
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  std::vector<Cache::Handle*> pins(10, nullptr);
  ASSERT_OK(Insert(cache.get(), c, kU, &pins[0]));
  for (uint32_t i = 1; i < 10; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU, &pins[i]));
  }
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  ASSERT_TRUE(fifo->TEST_InGhost(K("a")));
  Cache::Handle* h = nullptr;
  EXPECT_TRUE(Insert(cache.get(), K("a"), kU, &h).IsMemoryLimit());
  EXPECT_TRUE(fifo->TEST_InGhost(K("a")));
  // Evicts c, whose tag replaces a's, and finds nothing else to evict.
  cache->Release(pins[0]);
  EXPECT_TRUE(Insert(cache.get(), K("a"), 2 * kU, &h).IsMemoryLimit());
  EXPECT_FALSE(Holds(cache.get(), c));
  EXPECT_TRUE(fifo->TEST_InGhost(c));
  EXPECT_FALSE(fifo->TEST_InGhost(K("a")));
  for (uint32_t i = 1; i < 10; ++i) {
    cache->Release(pins[i]);
  }
}

// An insert that finds the table's occupancy full of pinned entries goes
// standalone, outside the table, and puts back the tag it took.
TEST_F(FIFOCacheTest, GhostTagPutBackWhenInsertGoesStandalone) {
  // The table can't grow past its 2048 slots, which hold 1229 entries.
  std::shared_ptr<Cache> cache = NewCache(4096);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a")));
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 0; i < 2000 && Holds(cache.get(), K("a")); ++i) {
    pins.push_back(nullptr);
    ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 1, &pins.back()));
  }
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  ASSERT_TRUE(fifo->TEST_InGhost(K("a")));
  Cache::Handle* h = nullptr;
  EXPECT_TRUE(Insert(cache.get(), K("a"), /*charge*/ 1, &h).IsOkOverwritten());
  EXPECT_TRUE(fifo->TEST_InGhost(K("a")));
  cache->Release(h);
  // One whose key wasn't in the ghost leaves no tag, at a home other than
  // a's.
  UniqueId64x2 hashed = Shard::ComputeHash(K("a"), cache->GetHashSeed());
  hashed[1] ^= 1;
  UniqueId64x2 unhashed;
  const std::string b =
      Shard::ReverseHash(hashed, &unhashed, cache->GetHashSeed()).ToString();
  ASSERT_FALSE(fifo->TEST_InGhost(b));
  EXPECT_TRUE(Insert(cache.get(), b, /*charge*/ 1, &h).IsOkOverwritten());
  EXPECT_FALSE(fifo->TEST_InGhost(b));
  cache->Release(h);
  for (Cache::Handle* pin : pins) {
    cache->Release(pin);
  }
  ASSERT_OK(Insert(cache.get(), K("a")));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
}

// A key readmitted from the ghost starts at no hits with a handle too: the
// release of the reference Insert returned isn't a hit. With probation
// under its share the resident clock pays for the next insert, and takes it
// before entries hit once, whatever order the clock visits them in.
TEST_F(FIFOCacheTest, GhostReadmittedWithHandleStartsAtNoHits) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  const std::vector<std::string> nine = NineKeysWithOwnHomes(cache.get());
  const std::string a = nine[8];
  ASSERT_NO_FATAL_FAILURE(GhostNineBehindBig(cache.get(), fifo, nine));
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK(Insert(cache.get(), nine[i], kU));
    ASSERT_FALSE(fifo->TEST_InProbation(nine[i]));
    HitNTimes(cache.get(), nine[i], 1);
  }
  Cache::Handle* h = nullptr;
  ASSERT_OK(Insert(cache.get(), a, kU, &h));
  ASSERT_FALSE(fifo->TEST_InProbation(a));
  cache->Release(h);
  ASSERT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K(8), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K(8)));
  EXPECT_FALSE(Holds(cache.get(), a));
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(Holds(cache.get(), nine[i])) << i;
  }
  EXPECT_TRUE(Holds(cache.get(), K("big")));
}

// The clock takes the insert reference's count out once: keys readmitted
// with a handle and hit twice keep both hits, and outlive one hit once.
TEST_F(FIFOCacheTest, GhostReadmittedWithHandleKeepsItsHits) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  const std::vector<std::string> nine = NineKeysWithOwnHomes(cache.get());
  const std::string z = nine[8];
  ASSERT_NO_FATAL_FAILURE(GhostNineBehindBig(cache.get(), fifo, nine));
  for (int i = 0; i < 8; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), nine[i], kU, &h));
    ASSERT_FALSE(fifo->TEST_InProbation(nine[i]));
    cache->Release(h);
    HitNTimes(cache.get(), nine[i], 2);
  }
  ASSERT_OK(Insert(cache.get(), z, kU));
  ASSERT_FALSE(fifo->TEST_InProbation(z));
  HitNTimes(cache.get(), z, 1);
  ASSERT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K(8), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K(8)));
  EXPECT_FALSE(Holds(cache.get(), z));
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(Holds(cache.get(), nine[i])) << i;
  }
  EXPECT_TRUE(Holds(cache.get(), K("big")));
}

// While the reference Insert returned is held the readmitted entry is
// pinned, so the clock passes it and leaves its counters alone; once
// released, the clock takes the reference's count out.
TEST_F(FIFOCacheTest, GhostReadmittedEntryPinnedThroughTheClock) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  const std::vector<std::string> nine = NineKeysWithOwnHomes(cache.get());
  const std::string a = nine[8];
  ASSERT_NO_FATAL_FAILURE(GhostNineBehindBig(cache.get(), fifo, nine));
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK(Insert(cache.get(), nine[i], kU));
    HitNTimes(cache.get(), nine[i], 2);
  }
  Cache::Handle* h = nullptr;
  ASSERT_OK(Insert(cache.get(), a, kU, &h));
  ASSERT_FALSE(fifo->TEST_InProbation(a));
  // The clock passes a, pinned, and takes others after two laps.
  ASSERT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  ASSERT_OK(Insert(cache.get(), K(8), kU));
  ASSERT_FALSE(fifo->TEST_InProbation(K(8)));
  EXPECT_TRUE(Holds(cache.get(), a));
  EXPECT_EQ(kU, cache->GetPinnedUsage());
  cache->Release(h);
  EXPECT_EQ(0, cache->GetPinnedUsage());
  const size_t usage = cache->GetUsage();
  cache->Erase(a);
  EXPECT_EQ(usage - kU, cache->GetUsage());
  EXPECT_FALSE(Holds(cache.get(), a));
}

// A placeholder insert (no charge, kNoopCacheItemHelper), which
// CacheWithSecondaryAdapter uses to note a recent use, neither takes a
// ghost nor leaves one, so readmission waits for the real entry, which
// replaces the placeholder.
TEST_F(FIFOCacheTest, GhostIgnoresPlaceholderInserts) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  for (uint32_t i = 0; i < 9; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // Evicts a.
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_TRUE(fifo->TEST_InGhost(K("a")));
  ASSERT_OK(Insert(cache.get(), K("a"), /*charge*/ 0));
  EXPECT_TRUE(fifo->TEST_InProbation(K("a")));
  EXPECT_TRUE(fifo->TEST_InGhost(K("a")));
  Status s = Insert(cache.get(), K("a"), kU);
  EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
  EXPECT_FALSE(fifo->TEST_InGhost(K("a")));
  Cache::Handle* h = cache->Lookup(K("a"));
  ASSERT_NE(nullptr, h);
  EXPECT_EQ(kU, cache->GetCharge(h));
  cache->Release(h);
  EXPECT_EQ(1, Keys(cache.get()).count(K("a")));
  // A placeholder evicted from probation isn't remembered.
  ASSERT_OK(Insert(cache.get(), K("p"), /*charge*/ 0));
  for (uint32_t i = 100; i < 110; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_FALSE(Holds(cache.get(), K("p")));
  EXPECT_FALSE(fifo->TEST_InGhost(K("p")));
}

// End to end through the Cache interface: a key read once per scan longer
// than the cache misses every time without the ghost. With it, the second
// miss lands in the resident queue and every later read hits despite the
// scans.
TEST_F(FIFOCacheTest, GhostKeepsSlowReuseKeyAcrossScans) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  auto access = [&](const std::string& key) {
    if (Cache::Handle* h = cache->Lookup(key)) {
      cache->Release(h);
      return true;
    }
    EXPECT_OK(Insert(cache.get(), key, kU));
    return false;
  };
  uint32_t scanned = 1000;
  auto scan = [&] {
    for (int i = 0; i < 150; ++i) {
      EXPECT_FALSE(access(K(scanned++)));
    }
  };
  EXPECT_FALSE(access(K("hot")));
  scan();
  // The scan evicted hot first and 50 more after it. The ghost is
  // direct-mapped, so one of those whose key has hot's home would have
  // replaced its tag; with this hash seed none does.
  ASSERT_TRUE(fifo->TEST_InGhost(K("hot")));
  EXPECT_FALSE(access(K("hot")));
  EXPECT_FALSE(fifo->TEST_InProbation(K("hot")));
  scan();
  for (int round = 2; round < 5; ++round) {
    EXPECT_TRUE(access(K("hot"))) << round;
    scan();
  }
}

// A probation victim larger than probation's share is remembered and
// readmitted, then survives a cold stream of mixed charges.
TEST_F(FIFOCacheTest, GhostReadmitsOversizedVictim) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a"), 20 * kU));
  for (uint32_t i = 0; i < 80; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_OK(Insert(cache.get(), K("x"), kU));
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  ASSERT_OK(Insert(cache.get(), K("a"), 20 * kU));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
  for (uint32_t i = 0; i < 200; ++i) {
    ASSERT_OK(Insert(cache.get(), K(1000 + i), (i % 2 ? 15 : 1) * kU));
    ASSERT_LE(cache->GetUsage(), 100 * kU);
  }
  EXPECT_TRUE(Holds(cache.get(), K("a")));
}

// Ghost tags are found at a key's home after the table grows: each split
// copies the tag at the home it splits from to the new one.
TEST_F(FIFOCacheTest, GhostTagsSurviveTableGrowth) {
  // Reserved for 1 MiB, so the table can grow well past its 64 slots.
  std::shared_ptr<Cache> cache = NewCache(1 << 20);
  FIFOCache* fifo = Fifo(cache.get());
  cache->SetCapacity(10 * kU);
  for (uint32_t i = 0; i < 20; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  // K(0)..K(9) were evicted; those whose tags another didn't overwrite are
  // in the ghost.
  std::vector<std::string> ghosts;
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_FALSE(Holds(cache.get(), K(i)));
    if (fifo->TEST_InGhost(K(i))) {
      ghosts.push_back(K(i));
    }
  }
  ASSERT_GE(ghosts.size(), 5);
  const size_t slots = cache->GetTableAddressCount();
  cache->SetCapacity(1 << 20);
  for (uint32_t i = 1000; i < 2000; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  ASSERT_GE(cache->GetTableAddressCount(), 16 * slots);
  for (const std::string& key : ghosts) {
    EXPECT_TRUE(fifo->TEST_InGhost(key));
  }
  ASSERT_OK(Insert(cache.get(), ghosts[0], kU));
  EXPECT_FALSE(fifo->TEST_InProbation(ghosts[0]));
}

// A ghost tag is never 0, the empty value, so a key whose hash has zeros
// in the 32 bits the tag comes from isn't found in an empty ghost.
TEST_F(FIFOCacheTest, KeyWithZeroTagBitsIsNotInEmptyGhost) {
  std::shared_ptr<Cache> cache = NewCache(10 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  const UniqueId64x2 hashed{uint64_t{0xabcd} << 32, 0x1234};
  UniqueId64x2 unhashed;
  const std::string key =
      Shard::ReverseHash(hashed, &unhashed, cache->GetHashSeed()).ToString();
  EXPECT_FALSE(fifo->TEST_InGhost(key));
  ASSERT_OK(Insert(cache.get(), key, kU));
  EXPECT_TRUE(fifo->TEST_InProbation(key));
}

// A fifo_cache:// string builds FIFOCache, not a fallback: a probation
// victim inserted again is readmitted to the resident queue.
TEST_F(FIFOCacheTest, CreateFromStringPreservesSelectedBehavior) {
  std::shared_ptr<Cache> cache;
  ASSERT_OK(Cache::CreateFromString(
      ConfigOptions(),
      "fifo_cache://capacity=" + std::to_string(10 * kU) +
          ";num_shard_bits=0;hash_seed=12345;"
          "metadata_charge_policy=kDontChargeCacheMetadata",
      &cache));
  EXPECT_STREQ("FIFOCache", cache->Name());
  FIFOCache* fifo = dynamic_cast<FIFOCache*>(cache.get());
  ASSERT_NE(nullptr, fifo);
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_FALSE(Holds(cache.get(), K("a")));
  ASSERT_OK(Insert(cache.get(), K("a"), kU));
  EXPECT_FALSE(fifo->TEST_InProbation(K("a")));
}

#ifndef NDEBUG
namespace {
// Runs op on another thread, parks it the first time it reaches sync point
// `point`, runs `meanwhile` here, then lets op finish.
void RunWhileParked(const std::string& point, const std::function<void()>& op,
                    const std::function<void()>& meanwhile) {
  std::promise<void> reached;
  std::promise<void> resume;
  std::shared_future<void> resumed = resume.get_future().share();
  std::atomic<bool> parked{false};
  SyncPoint::GetInstance()->SetCallBack(point, [&](void*) {
    if (!parked.exchange(true)) {
      reached.set_value();
      resumed.wait();
    }
  });
  SyncPoint::GetInstance()->EnableProcessing();
  std::promise<void> finished;
  std::future<void> op_finished = finished.get_future();
  port::Thread t([&] {
    op();
    finished.set_value();
  });
  // Fails rather than hangs if op finishes without reaching the point.
  std::future<void> op_reached = reached.get_future();
  while (op_reached.wait_for(std::chrono::milliseconds(1)) !=
         std::future_status::ready) {
    if (op_finished.wait_for(std::chrono::seconds(0)) ==
        std::future_status::ready) {
      break;
    }
  }
  if (op_reached.wait_for(std::chrono::seconds(0)) ==
      std::future_status::ready) {
    meanwhile();
  } else {
    ADD_FAILURE() << point << " was not reached";
  }
  resume.set_value();
  t.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
}
}  // namespace

// An entry erased after it is linked and before it is queued leaves
// probation's usage at once, and the cell queued after it is stale:
// eviction passes over it, to the entry behind.
TEST_F(FIFOCacheTest, EraseBeforeProbationPush) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  RunWhileParked(
      "FIFOCache::DoInsert:BeforeProbationPush",
      [&] { ASSERT_OK(Insert(cache.get(), K("k"), 2)); },
      [&] {
        EXPECT_EQ(2, fifo->TEST_GetProbationUsage());
        cache->Erase(K("k"));
        EXPECT_EQ(0, fifo->TEST_GetProbationUsage());
        EXPECT_EQ(0, cache->GetUsage());
        ASSERT_OK(Insert(cache.get(), K("a")));
      });
  // Queued: a, then k's stale cell.
  ASSERT_OK(Insert(cache.get(), K("b")));
  EXPECT_EQ(2, fifo->TEST_GetProbationUsage());
  for (const char* key : {"c", "d", "e"}) {
    ASSERT_OK(Insert(cache.get(), K(key)));
  }
  EXPECT_EQ((std::vector<std::string>{K("a"), K("b"), K("c")}), evicted);
  EXPECT_EQ(2, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(2, cache->GetUsage());
}

// Hits that land between an eviction step's read of the head and its claim
// on it make the claim fail, and the step decides again with them counted:
// a head pinned meanwhile rotates, one hit twice meanwhile is promoted, and
// either way the next head is evicted instead.
TEST_F(FIFOCacheTest, HitsBetweenProbationStepReadAndClaimCount) {
  for (bool pin : {true, false}) {
    SCOPED_TRACE(pin ? "pinned" : "hit twice");
    std::vector<std::string> evicted;
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
    RecordEvictions(cache.get(), &evicted);
    ASSERT_OK(Insert(cache.get(), K("a")));
    ASSERT_OK(Insert(cache.get(), K("b")));
    Cache::Handle* ha = nullptr;
    RunWhileParked(
        "FIFOCache::ProbationStep:Loaded",
        [&] { ASSERT_OK(Insert(cache.get(), K("c"))); },
        [&] {
          if (pin) {
            ha = cache->Lookup(K("a"));
            ASSERT_NE(nullptr, ha);
          } else {
            HitNTimes(cache.get(), K("a"), 2);
          }
        });
    EXPECT_EQ(std::vector<std::string>{K("b")}, evicted);
    EXPECT_TRUE(Holds(cache.get(), K("a")));
    EXPECT_TRUE(Holds(cache.get(), K("c")));
    if (ha != nullptr) {
      EXPECT_FALSE(cache->Release(ha));
    }
  }
}

// Probation heads that lose every claim to hits rotate without freeing
// anything or counting toward the effort cap, so the per-call limit on
// probation steps is what hands the eviction to the resident clock.
TEST_F(FIFOCacheTest, HeadsLosingEveryClaimYieldToResidentClock) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 4);
  RecordEvictions(cache.get(), &evicted);
  for (const char* key : {"r1", "r2"}) {
    ASSERT_OK(Insert(cache.get(), K(key)));
    HitNTimes(cache.get(), K(key), 2);
  }
  ASSERT_OK(Insert(cache.get(), K("a")));
  ASSERT_OK(Insert(cache.get(), K("b")));
  // Promotes r1 and r2 and evicts a, leaving b and c in probation.
  ASSERT_OK(Insert(cache.get(), K("c")));
  ASSERT_EQ(std::vector<std::string>{K("a")}, evicted);
  size_t claims_lost = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "FIFOCache::ProbationStep:Loaded", [&](void*) {
        // Stops interfering eventually, so a missing limit fails the test
        // rather than hanging it.
        if (++claims_lost > 1000) {
          return;
        }
        for (const char* key : {"b", "c"}) {
          if (Cache::Handle* h = cache->Lookup(K(key))) {
            cache->Release(h);
          }
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Insert(cache.get(), K("d")));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  // The clock's step covers both resident entries, and takes both.
  ASSERT_EQ(3, evicted.size());
  EXPECT_EQ((std::set<std::string>{K("r1"), K("r2")}),
            (std::set<std::string>{evicted[1], evicted[2]}));
  EXPECT_TRUE(Holds(cache.get(), K("b")));
  EXPECT_TRUE(Holds(cache.get(), K("c")));
  EXPECT_TRUE(Holds(cache.get(), K("d")));
  EXPECT_GT(claims_lost, 4);
  EXPECT_LT(claims_lost, 1000);
}

// With nothing in the resident queue, eviction goes to probation even when
// probation is within its share (here because nearly all of it is pinned),
// without running the clock over a table it can take nothing from.
TEST_F(FIFOCacheTest, EmptyResidentQueueSkipsTheClock) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  std::vector<Cache::Handle*> pins;
  for (uint32_t i = 200; i < 223; ++i) {
    Cache::Handle* h = nullptr;
    ASSERT_OK(Insert(cache.get(), K(i), i < 222 ? 4 * kU : kU, &h));
    pins.push_back(h);
  }
  for (uint32_t i = 0; i < 11; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), kU));
  }
  ASSERT_EQ(100 * kU, cache->GetUsage());
  // Over its share: the pins rotate and K(0) is evicted.
  ASSERT_OK(Insert(cache.get(), K("y"), kU));
  ASSERT_EQ(std::vector<std::string>{K(0)}, evicted);
  ASSERT_EQ(89 * kU, fifo->TEST_GetProbationPinnedUsage());
  // Probation's 11 unpinned plus z's 1 make its share of 12.
  size_t resident_steps = 0;
  SyncPoint::GetInstance()->SetCallBack("FIFOCache::Evict:ResidentStep",
                                        [&](void*) { ++resident_steps; });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Insert(cache.get(), K("z"), kU));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  EXPECT_EQ((std::vector<std::string>{K(0), K(1)}), evicted);
  EXPECT_EQ(0, resident_steps);
  for (Cache::Handle* h : pins) {
    EXPECT_FALSE(cache->Release(h));
  }
}

// A hit between the clock's read of a readmitted entry's counters and its
// taking the insert reference's count out makes the clock leave that for a
// later visit, rather than drop the bit with the count still in: the entry,
// hit once, goes before entries hit twice.
TEST_F(FIFOCacheTest, HitWhileClockDropsInsertRefCount) {
  std::shared_ptr<Cache> cache = NewCache(100 * kU);
  FIFOCache* fifo = Fifo(cache.get());
  const std::vector<std::string> nine = NineKeysWithOwnHomes(cache.get());
  const std::string a = nine[8];
  ASSERT_NO_FATAL_FAILURE(GhostNineBehindBig(cache.get(), fifo, nine));
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK(Insert(cache.get(), nine[i], kU));
    HitNTimes(cache.get(), nine[i], 2);
  }
  Cache::Handle* h = nullptr;
  ASSERT_OK(Insert(cache.get(), a, kU, &h));
  ASSERT_FALSE(fifo->TEST_InProbation(a));
  cache->Release(h);
  ASSERT_EQ(12 * kU, fifo->TEST_GetProbationUsage());
  RunWhileParked(
      "FIFOCache::DropInsertRefCount:Loaded",
      [&] { ASSERT_OK(Insert(cache.get(), K(8), kU)); },
      [&] { HitNTimes(cache.get(), a, 1); });
  EXPECT_FALSE(Holds(cache.get(), a));
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(Holds(cache.get(), nine[i])) << i;
  }
}

// An entry erased while its insert still holds the reference it links the
// entry with is left invisible with no references, as in HyperClockCache,
// until a probation step frees it; being erased, it leaves no ghost.
TEST_F(FIFOCacheTest, ErasedEntryFreedByEvictionLeavesNoGhost) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
  FIFOCache* fifo = Fifo(cache.get());
  RunWhileParked(
      "FIFOCache::DoInsert:Linked",
      [&] { ASSERT_OK(Insert(cache.get(), K("k"))); },
      [&] { cache->Erase(K("k")); });
  EXPECT_FALSE(Holds(cache.get(), K("k")));
  EXPECT_EQ(1, cache->GetUsage());
  ASSERT_OK(Insert(cache.get(), K("a")));
  ASSERT_OK(Insert(cache.get(), K("b")));
  EXPECT_EQ(2, cache->GetUsage());
  EXPECT_TRUE(Holds(cache.get(), K("a")));
  EXPECT_FALSE(fifo->TEST_InGhost(K("k")));
}

// An insert whose entry is replaced by a newer insert of the key between
// linking it and erasing the older entries stops there, leaving the newer
// entry, which erased the rest.
TEST_F(FIFOCacheTest, ReplacedInsertLeavesNewerEntry) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 10 * kU);
  ASSERT_OK(cache->Insert(K("k"), new CountedValue(1), CountedHelper(), kU));
  RunWhileParked(
      "FIFOCache::DoInsert:Linked",
      [&] {
        ASSERT_OK(
            cache->Insert(K("k"), new CountedValue(2), CountedHelper(), kU));
      },
      [&] {
        Status s =
            cache->Insert(K("k"), new CountedValue(3), CountedHelper(), kU);
        EXPECT_TRUE(s.IsOkOverwritten()) << s.ToString();
      });
  Cache::Handle* h = cache->Lookup(K("k"));
  ASSERT_NE(nullptr, h);
  CheckCountedValue(cache.get(), h, 3);
  cache->Release(h);
  EXPECT_EQ((std::set<std::string>{K("k")}), Keys(cache.get()));
  cache.reset();
  EXPECT_EQ(0, LiveValues().load());
}

// An insert that charges a ring and then finds another insert added it
// first takes its charge back: usage stays the entries' charges and the
// table's metadata, with one ring added.
TEST_F(FIFOCacheTest, RingChargeTakenBackWhenAnotherInsertAddsIt) {
  std::shared_ptr<Cache> cache =
      NewCache(1 << 20, /*strict*/ false, kFullChargeCacheMetadata);
  FIFOCache* fifo = Fifo(cache.get());
  for (uint32_t i = 0; i < 64; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  const size_t cells_before = (fifo->TEST_GetMetadataCharge() -
                               cache->GetTableAddressCount() * kSlotCharge) /
                              kQueueCellCharge;
  ASSERT_EQ(64, cells_before);
  RunWhileParked(
      "FIFOCache::ProbationPush:RingCharged",
      [&] { ASSERT_OK(Insert(cache.get(), K(64))); },
      [&] { ASSERT_OK(Insert(cache.get(), K(65))); });
  EXPECT_EQ(64 + 128, (fifo->TEST_GetMetadataCharge() -
                       cache->GetTableAddressCount() * kSlotCharge) /
                          kQueueCellCharge);
  EXPECT_EQ(66 + fifo->TEST_GetMetadataCharge(), cache->GetUsage());
  EXPECT_TRUE(fifo->TEST_InProbation(K(64)));
  EXPECT_TRUE(fifo->TEST_InProbation(K(65)));
}

// An entry erased while a step promotes it (holding a reference, so it can
// read the charge) leaves probation's usage, and the step's release of that
// last reference frees it.
TEST_F(FIFOCacheTest, EraseWhilePromotionHoldsItsReference) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 4);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  ASSERT_OK(Insert(cache.get(), K("a"), 2));
  HitNTimes(cache.get(), K("a"), 2);
  ASSERT_OK(Insert(cache.get(), K("b"), 2));
  RunWhileParked(
      "FIFOCache::ProbationStep:Promoting",
      [&] { ASSERT_OK(Insert(cache.get(), K("c"))); },
      [&] {
        cache->Erase(K("a"));
        EXPECT_EQ(nullptr, cache->Lookup(K("a")));
        EXPECT_EQ(4, cache->GetUsage());
      });
  EXPECT_EQ(std::vector<std::string>{K("b")}, evicted);
  EXPECT_FALSE(Holds(cache.get(), K("a")));
  EXPECT_EQ(1, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(1, cache->GetUsage());
}

// The same when a step holds a reference to mark a pinned head: the head's
// holder lets go after it is erased, and the step's release frees it.
TEST_F(FIFOCacheTest, EraseWhileStepMarksHeadPinned) {
  std::vector<std::string> evicted;
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
  RecordEvictions(cache.get(), &evicted);
  FIFOCache* fifo = Fifo(cache.get());
  Cache::Handle* ha = nullptr;
  ASSERT_OK(Insert(cache.get(), K("a"), 1, &ha));
  ASSERT_OK(Insert(cache.get(), K("b")));
  RunWhileParked(
      "FIFOCache::WithReference:Held",
      [&] { ASSERT_OK(Insert(cache.get(), K("c"))); },
      [&] {
        cache->Erase(K("a"));
        EXPECT_FALSE(cache->Release(ha));
        EXPECT_EQ(2, cache->GetUsage());
      });
  // The step went on to evict b for c.
  EXPECT_EQ(std::vector<std::string>{K("b")}, evicted);
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_EQ(1, fifo->TEST_GetProbationUsage());
  EXPECT_EQ(0, fifo->TEST_GetProbationPinnedUsage());
}

// A step's claim on a head takes only the entry its cell was queued for.
// Erasing a key and inserting it again while a step is between reading the
// head and claiming it puts a new entry in the same slot with the same
// counters; the step leaves it alone (its cell is stale) and evicts the next
// head, and the new entry keeps its hits.
TEST_F(FIFOCacheTest, StepClaimsOnlyTheEntryItsCellWasQueuedFor) {
  for (bool hit : {false, true}) {
    SCOPED_TRACE(hit ? "hit twice" : "no hits");
    std::vector<std::string> evicted;
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
    RecordEvictions(cache.get(), &evicted);
    ASSERT_OK(Insert(cache.get(), K("a")));
    if (hit) {
      HitNTimes(cache.get(), K("a"), 2);
    }
    ASSERT_OK(Insert(cache.get(), K("b")));
    RunWhileParked(
        "FIFOCache::ProbationStep:Loaded",
        [&] { ASSERT_OK(Insert(cache.get(), K("c"))); },
        [&] {
          cache->Erase(K("a"));
          ASSERT_OK(Insert(cache.get(), K("a")));
          if (hit) {
            HitNTimes(cache.get(), K("a"), 2);
          }
        });
    EXPECT_EQ(std::vector<std::string>{K("b")}, evicted);
    // Queued: the new a, then c. Hit twice, a is promoted and c goes;
    // otherwise a goes.
    ASSERT_OK(Insert(cache.get(), K("d")));
    EXPECT_EQ((std::vector<std::string>{K("b"), hit ? K("c") : K("a")}),
              evicted);
  }
}

// Parks the first thread to reach a sync point until Resume. Set processing
// up after constructing, and clear the callbacks before it goes.
class Parker {
 public:
  explicit Parker(const std::string& point) {
    SyncPoint::GetInstance()->SetCallBack(point, [this](void*) {
      if (!parked_.exchange(true)) {
        reached_.set_value();
        resumed_.wait();
      }
    });
  }
  void WaitParked() { reached_.get_future().wait(); }
  void Resume() { resume_.set_value(); }

 private:
  std::atomic<bool> parked_{false};
  std::promise<void> reached_;
  std::promise<void> resume_;
  std::shared_future<void> resumed_ = resume_.get_future().share();
};

// A pop that empties a ring with a newer one after it closes the ring before
// moving on, in case the thread adding the newer ring hasn't yet. A push
// that read the old ring as the newest then goes to the newer ring, rather
// than into a ring no pop looks at again.
TEST_F(FIFOCacheTest, ProbationQueuePopClosesDrainedRing) {
  fifo_cache::ProbationQueue q(/*first_ring_capacity*/ 2,
                               /*max_ring_capacity*/ 4);
  ASSERT_TRUE(q.Push(10));
  Parker pusher("FIFOCache::ProbationQueue::Push:Loaded");
  Parker adder("FIFOCache::ProbationQueue::TryAddRing:BeforeClose");
  SyncPoint::GetInstance()->EnableProcessing();
  port::Thread push_thread([&] { EXPECT_TRUE(q.Push(20)); });
  pusher.WaitParked();
  port::Thread add_thread([&] { EXPECT_TRUE(q.TryAddRing(4)); });
  adder.WaitParked();
  uint64_t v = 0;
  EXPECT_TRUE(q.Pop(&v));
  EXPECT_EQ(10, v);
  EXPECT_FALSE(q.Pop(&v));
  pusher.Resume();
  push_thread.join();
  adder.Resume();
  add_thread.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_TRUE(q.Pop(&v));
  EXPECT_EQ(20, v);
  EXPECT_FALSE(q.Pop(&v));
}

// A pop doesn't move past a closed ring while a push to it has claimed a
// cell and not yet filled it.
TEST_F(FIFOCacheTest, ProbationQueuePopWaitsForPushInFlight) {
  fifo_cache::ProbationQueue q(/*first_ring_capacity*/ 2,
                               /*max_ring_capacity*/ 4);
  Parker pusher("FIFOCache::ProbationQueue::Push:Claimed");
  SyncPoint::GetInstance()->EnableProcessing();
  port::Thread push_thread([&] { EXPECT_TRUE(q.Push(10)); });
  pusher.WaitParked();
  EXPECT_TRUE(q.TryAddRing(4));
  uint64_t v = 0;
  EXPECT_FALSE(q.Pop(&v));
  pusher.Resume();
  push_thread.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_TRUE(q.Pop(&v));
  EXPECT_EQ(10, v);
}

// In a shard whose entries are all pinned, or all but one percent, an
// insert's eviction stops at the effort cap (eviction_effort_cap, 30 pinned
// entries seen per entry freed plus 30, so 60 for an insert that frees the
// one entry it needs) instead of passing every pinned entry, and the insert
// goes over capacity, as in HyperClockCache. A probation step sees one
// entry and a resident step the entries of 4 homes, some with none pinned,
// so kBound steps leaves room for those while staying far below a lap of
// either queue at 16000 entries (16000 probation cells, about 6700
// resident steps).
// An insert that charges a probation ring evicts for the ring only, not
// for usage something else took over capacity: here a standalone entry
// larger than everything cached (so it evicts nothing for itself) takes
// usage far over capacity while an insert is between charging a ring and
// adding it. That insert evicts about the ring's bytes, and the rest is
// left to later inserts, as before.
TEST_F(FIFOCacheTest, ChargedRingEvictsOnlyForTheRing) {
  constexpr size_t kCapacity = 64 << 10;
  std::shared_ptr<Cache> cache =
      NewCache(kCapacity, /*strict*/ false, kFullChargeCacheMetadata);
  FIFOCache* fifo = Fifo(cache.get());
  for (uint32_t i = 0; i < 200; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i), 1000));
  }
  std::atomic<bool> stop{false};
  bool parked = false;
  size_t occupancy = 0;
  size_t metadata = 0;
  Cache::Handle* big = nullptr;
  RunWhileParked(
      "FIFOCache::ProbationPush:RingCharged",
      [&] {
        for (uint32_t i = 200; i < 2000 && !stop.load(); ++i) {
          ASSERT_OK(Insert(cache.get(), K(i), 100));
        }
      },
      [&] {
        parked = true;
        occupancy = cache->GetOccupancyCount();
        metadata = fifo->TEST_GetMetadataCharge();
        big = cache->CreateStandalone("big", nullptr, &kNoopCacheItemHelper,
                                      4 * kCapacity,
                                      /*allow_uncharged*/ false);
        stop = true;
      });
  ASSERT_TRUE(parked);
  ASSERT_NE(nullptr, big);
  const size_t ring = fifo->TEST_GetMetadataCharge() - metadata;
  EXPECT_GT(ring, 0);
  // Every entry is charged at least 100.
  EXPECT_LE(occupancy, cache->GetOccupancyCount() + ring / 100 + 1);
  EXPECT_GT(cache->GetUsage(), 4 * kCapacity);
  EXPECT_TRUE(cache->Release(big));
}

TEST_F(FIFOCacheTest, MostlyPinnedEvictionVisitsBounded) {
  constexpr size_t kBound = 256;
  // Large enough that 16000 entries fit the table's reservation.
  constexpr size_t kC = 128;
  size_t steps = 0;
  SyncPoint::GetInstance()->SetCallBack("FIFOCache::ProbationStep:Loaded",
                                        [&](void*) { ++steps; });
  SyncPoint::GetInstance()->SetCallBack("FIFOCache::Evict:ResidentStep",
                                        [&](void*) { ++steps; });
  for (uint32_t n : {1000u, 16000u}) {
    for (bool all_pinned : {true, false}) {
      SCOPED_TRACE(std::to_string(n) + (all_pinned ? " all" : " 99%"));
      std::shared_ptr<Cache> cache = NewCache(n * kC);
      std::vector<Cache::Handle*> pins;
      if (all_pinned) {
        // Half promoted to the resident queue, half in probation.
        for (uint32_t i = 0; i < n / 2; ++i) {
          ASSERT_OK(Insert(cache.get(), "r" + std::to_string(i), kC));
          HitNTimes(cache.get(), "r" + std::to_string(i), 2);
        }
        for (uint32_t i = 0; i < n / 2; ++i) {
          ASSERT_OK(Insert(cache.get(), "p" + std::to_string(i), kC));
        }
        ASSERT_OK(Insert(cache.get(), "x", kC));
        for (const std::string& key : Keys(cache.get())) {
          pins.push_back(cache->Lookup(key));
          ASSERT_NE(nullptr, pins.back());
        }
        ASSERT_EQ(n, pins.size());
      } else {
        for (uint32_t i = 0; i < n - n / 100; ++i) {
          Cache::Handle* h = nullptr;
          ASSERT_OK(Insert(cache.get(), "p" + std::to_string(i), kC, &h));
          pins.push_back(h);
        }
        for (uint32_t i = 0; i < n / 100; ++i) {
          ASSERT_OK(Insert(cache.get(), "u" + std::to_string(i), kC));
        }
      }
      ASSERT_EQ(n, cache->GetOccupancyCount());
      ASSERT_OK(Insert(cache.get(), "first", kC));
      size_t most = 0;
      SyncPoint::GetInstance()->EnableProcessing();
      for (uint32_t i = 0; i < 100; ++i) {
        steps = 0;
        ASSERT_OK(Insert(cache.get(), "y" + std::to_string(i), kC));
        most = std::max(most, steps);
      }
      SyncPoint::GetInstance()->DisableProcessing();
      EXPECT_LE(most, kBound);
      EXPECT_TRUE(Holds(cache.get(), "y99"));
      EXPECT_EQ(pins.size() * kC, cache->GetPinnedUsage());
      for (Cache::Handle* h : pins) {
        cache->Release(h);
      }
    }
  }
  SyncPoint::GetInstance()->ClearAllCallBacks();
}
#endif  // !NDEBUG

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
