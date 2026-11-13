//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "plugin/fifo_cache/fifo_cache.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cache/lru_cache.h"
#include "port/port.h"
#include "rocksdb/cache.h"
#include "rocksdb/convenience.h"
#include "rocksdb/table.h"
#include "test_util/testharness.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace {

// Block cache keys are 16 bytes, the only length the cache takes for now.
// Names up to 16 bytes, padded with zeros.
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

// Each slot of the table is 64 bytes, charged under kFullChargeCacheMetadata.
constexpr size_t kSlotSize = 64;

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
      int num_shard_bits = 0) {
    FIFOCacheOptions opts(capacity, num_shard_bits, strict_capacity_limit,
                          /*memory_allocator*/ nullptr, kDefaultToAdaptiveMutex,
                          policy);
    opts.hash_seed = 12345;
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
          const size_t slots_charge =
              policy == kFullChargeCacheMetadata
                  ? cache->GetTableAddressCount() * kSlotSize
                  : 0;
          EXPECT_LE(cache->GetUsage(), std::max(c.capacity, slots_charge) + 1);
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

TEST_F(FIFOCacheTest, UsageAndPinnedUsageAreExact) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 100);
  ASSERT_OK(Insert(cache.get(), K("a"), /*charge*/ 5));
  Cache::Handle* hb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("b"), /*charge*/ 7, &hb));
  ASSERT_NE(nullptr, hb);
  EXPECT_EQ(12, cache->GetUsage());
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
  EXPECT_EQ(12, cache->GetUsage());
  EXPECT_TRUE(cache->Release(hb));
  EXPECT_EQ(5, cache->GetUsage());
  EXPECT_EQ(0, cache->GetPinnedUsage());
}

// Under kFullChargeCacheMetadata usage counts every slot the table has grown
// to, besides the entries' charges; under kDontChargeCacheMetadata only the
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
                                  ? cache->GetTableAddressCount() * kSlotSize
                                  : 0;
      ASSERT_EQ(charges + metadata, cache->GetUsage()) << i;
    }
    EXPECT_GT(cache->GetTableAddressCount(), initial_slots);
    EXPECT_EQ(1000, cache->GetOccupancyCount());
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
// of failing, with or without a handle, as in HyperClockCache. Releasing
// the pins lets later inserts evict back under it.
TEST_F(FIFOCacheTest, NonStrictInsertOverFullyPinnedShardExceedsCapacity) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 2);
  Cache::Handle* pa = nullptr;
  Cache::Handle* pb = nullptr;
  ASSERT_OK(Insert(cache.get(), K("a"), 1, &pa));
  ASSERT_OK(Insert(cache.get(), K("b"), 1, &pb));
  ASSERT_OK(Insert(cache.get(), K("c")));
  Cache::Handle* pd = nullptr;
  ASSERT_OK(Insert(cache.get(), K("d"), 1, &pd));
  ASSERT_NE(nullptr, pd);
  EXPECT_EQ(4, cache->GetUsage());
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

// A 16-byte key is stored as a reversible hash; ApplyToAllEntries and the
// eviction callback give back the exact key, under a nonzero seed and on
// several shards.
TEST_F(FIFOCacheTest, KeysComeBackExact) {
  FIFOCacheOptions opts(/*capacity*/ 1 << 20, /*num_shard_bits*/ 2,
                        /*strict_capacity_limit*/ false, nullptr,
                        kDefaultToAdaptiveMutex, kDontChargeCacheMetadata);
  opts.hash_seed = 12345;
  std::shared_ptr<Cache> cache = opts.MakeSharedCache();
  ASSERT_NE(nullptr, cache);
  std::set<std::string> inserted;
  for (uint32_t i = 0; i < 200; ++i) {
    inserted.insert(K(i));
    ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 1000));
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
    inserted.insert(K(i));
    ASSERT_OK(Insert(cache.get(), K(i), /*charge*/ 1000));
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
// instead of the cache deleting it.
TEST_F(FIFOCacheTest, EvictionCallbackTakesValues) {
  std::vector<CountedValue*> taken;
  {
    std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 5);
    Cache* raw = cache.get();
    cache->SetEvictionCallback(
        [&taken, raw](const Slice& key, Cache::Handle* h, bool /*was_hit*/) {
          auto* value = static_cast<CountedValue*>(raw->Value(h));
          EXPECT_EQ(K(value->key), key.ToString());
          taken.push_back(value);
          return true;
        });
    ASSERT_OK(cache->Insert(K(0), new CountedValue(0), CountedHelper(), 1));
    cache->Erase(K(0));
    EXPECT_TRUE(taken.empty());
    for (uint32_t i = 1; i <= 20; ++i) {
      ASSERT_OK(cache->Insert(K(i), new CountedValue(i), CountedHelper(), 1));
    }
    EXPECT_GE(taken.size(), 15);
    for (CountedValue* value : taken) {
      EXPECT_EQ(kLiveMagic, value->magic.load());
    }
    const size_t evicted = taken.size();
    cache->EraseUnRefEntries();
    EXPECT_EQ(evicted, taken.size());
  }
  // The cache deleted everything but the values the callback took.
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

// Keys of any length but 16 bytes are refused: Insert returns NotSupported,
// Lookup and CreateStandalone find or make nothing, and Erase does nothing.
TEST_F(FIFOCacheTest, KeysOfOtherLengthsAreRefused) {
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 100);
  ASSERT_OK(Insert(cache.get(), K("a")));
  for (const std::string& key :
       {std::string(), std::string("a"), std::string(15, 'x'),
        std::string(17, 'x'), K("a") + "!"}) {
    SCOPED_TRACE(key.size());
    Cache::Handle* h = nullptr;
    Status s = Insert(cache.get(), key, 1, &h);
    EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
    EXPECT_EQ(nullptr, h);
    EXPECT_TRUE(Insert(cache.get(), key).IsNotSupported());
    EXPECT_EQ(nullptr, cache->Lookup(key));
    EXPECT_EQ(nullptr,
              cache->CreateStandalone(key, nullptr, &kNoopCacheItemHelper, 1,
                                      /*allow_uncharged*/ true));
    cache->Erase(key);
  }
  EXPECT_EQ(1, cache->GetUsage());
  EXPECT_EQ(1, cache->GetOccupancyCount());
  EXPECT_TRUE(Contains(cache.get(), K("a")));
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

// The table grows as entries arrive, and every key stays findable through
// growth and erasure.
TEST_F(FIFOCacheTest, TableGrowthKeepsEveryKeyFindable) {
  // Room in the table for every key.
  std::shared_ptr<Cache> cache = NewCache(/*capacity*/ 8 << 20);
  const size_t initial_slots = cache->GetTableAddressCount();
  constexpr uint32_t kKeys = 5000;
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_OK(Insert(cache.get(), K(i)));
  }
  EXPECT_GT(cache->GetTableAddressCount(), initial_slots);
  EXPECT_EQ(kKeys, cache->GetOccupancyCount());
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_TRUE(Contains(cache.get(), K(i))) << i;
  }
  for (uint32_t i = 0; i < kKeys; i += 3) {
    cache->Erase(K(i));
  }
  for (uint32_t i = 0; i < kKeys; ++i) {
    ASSERT_EQ(i % 3 != 0, Contains(cache.get(), K(i))) << i;
  }
}

// Inserts, lookups, erases, releases that erase, and capacity changes from
// several threads: every handle reads its own live value, and each value
// is deleted exactly once.
TEST_F(FIFOCacheTest, ConcurrentStressKeepsLifetimeExact) {
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
              Status s = cache->Insert(K(k), new CountedValue(k),
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
              cache->Erase(K(k));
              break;
            case 4:
              if (rnd.OneIn(100)) {
                cache->SetCapacity(rnd.OneIn(2) ? (16 << 10) : (64 << 10));
              }
              break;
            default: {
              Cache::Handle* h = cache->Lookup(K(k));
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
    // EraseUnRefEntries frees it. Afterwards only the slots are charged.
    const size_t slots_charge = cache->GetTableAddressCount() * kSlotSize;
    EXPECT_LE(charge_sum + slots_charge, cache->GetUsage());
    cache->EraseUnRefEntries();
    EXPECT_EQ(0, cache->GetOccupancyCount());
    EXPECT_EQ(slots_charge, cache->GetUsage());
  }
  EXPECT_EQ(0, LiveValues().load());
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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
