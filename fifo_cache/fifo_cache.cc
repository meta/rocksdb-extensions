//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "plugin/fifo_cache/fifo_cache.h"

#include <cinttypes>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cache/secondary_cache_adapter.h"
#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {
namespace fifo_cache {

namespace {
// A negative num_shard_bits gives each shard at least this much, or one
// shard for a smaller cache (fifo_cache_options.h says why 4 MiB).
constexpr size_t kMinShardSize = size_t{4} << 20;

// The table's options. The slice assignment sets every ShardedCacheOptions
// field, capacity included; estimated_entry_charge keeps its default of 0,
// which AutoHyperClockTable requires. use_adaptive_mutex has nothing to
// configure, and max_capacity goes to the cache's constructor beside these.
// num_shard_bits left negative gets shards of at least kMinShardSize.
HyperClockCacheOptions TableOptions(const FIFOCacheOptions& opts) {
  HyperClockCacheOptions table_opts(/*capacity*/ 0);
  static_cast<ShardedCacheOptions&>(table_opts) = opts;
  if (table_opts.num_shard_bits < 0) {
    table_opts.num_shard_bits =
        GetDefaultCacheShardBits(table_opts.capacity, kMinShardSize);
  }
  return table_opts;
}
}  // namespace

FIFOCache::FIFOCache(const FIFOCacheOptions& opts)
    : AutoHyperClockCache(TableOptions(opts), opts.max_capacity),
      max_capacity_(opts.max_capacity),
      metadata_charge_policy_(opts.metadata_charge_policy) {}

void FIFOCache::AppendPrintableOptions(std::string& str) const {
  AutoHyperClockCache::AppendPrintableOptions(str);
  str.append("    metadata_charge_policy : ");
  str.append(metadata_charge_policy_ == kFullChargeCacheMetadata
                 ? "kFullChargeCacheMetadata"
                 : "kDontChargeCacheMetadata");
  str.append("\n    max_capacity : ");
  str.append(std::to_string(max_capacity_));
  str.append("\n");
}

size_t FIFOCache::TEST_GetProbationUsage() const {
  return SumOverShards(
      [](Shard& shard) { return shard.GetTable().GetProbationUsage(); });
}

size_t FIFOCache::TEST_GetProbationPinnedUsage() const {
  return SumOverShards(
      [](Shard& shard) { return shard.GetTable().GetProbationPinnedUsage(); });
}

size_t FIFOCache::TEST_GetProbationTarget() const {
  return SumOverShards(
      [](Shard& shard) { return shard.GetTable().ProbationTarget(); });
}

bool FIFOCache::Release(Handle* handle, bool useful, bool erase_if_last_ref) {
  if (!useful) {
    auto* impl = reinterpret_cast<Shard::HandleImpl*>(handle);
    useful = GetShard(impl->GetHash()).GetTable().CountsInsertRef(impl);
  }
  return AutoHyperClockCache::Release(handle, useful, erase_if_last_ref);
}

bool FIFOCache::TEST_InProbation(const Slice& key) {
  Cache::Handle* h = Lookup(key);
  if (h == nullptr) {
    return false;
  }
  auto* impl = reinterpret_cast<Shard::HandleImpl*>(h);
  const bool in = GetShard(impl->GetHash()).GetTable().IsInProbation(impl);
  AutoHyperClockCache::Release(h, /*useful=*/false,
                               /*erase_if_last_ref=*/false);
  return in;
}

uint32_t FIFOCache::TEST_GetHits(const Slice& key) {
  Cache::Handle* h = Lookup(key);
  if (h == nullptr) {
    return 0;
  }
  auto* impl = reinterpret_cast<Shard::HandleImpl*>(h);
  const uint32_t hits = GetShard(impl->GetHash()).GetTable().HitsUnderRef(impl);
  AutoHyperClockCache::Release(h, /*useful=*/false,
                               /*erase_if_last_ref=*/false);
  return hits;
}

bool FIFOCache::TEST_InGhost(const Slice& key) const {
  const HashVal hash = Shard::ComputeHash(key, GetHashSeed());
  return const_cast<FIFOCache*>(this)->GetShard(hash).GetTable().InGhost(hash);
}

#ifndef NDEBUG
void FIFOCache::TEST_SetOccupancyLimit(size_t limit) {
  ForEachShard([limit](Shard* shard) {
    shard->GetTable().TEST_MutableOccupancyLimit() = limit;
  });
}
#endif  // !NDEBUG

size_t FIFOCache::TEST_GetMetadataCharge() const {
  return SumOverShards(
      [](Shard& shard) { return shard.GetTable().GetMetadataCharge(); });
}

FIFOCacheStats FIFOCache::GetStats() const {
  FIFOCacheStats stats;
  ForEachShard(
      [&stats](const Shard* shard) { shard->GetTable().AddStats(&stats); });
  return stats;
}

void FIFOCache::ReportProblems(const std::shared_ptr<Logger>& info_log) const {
  if (info_log == nullptr) {
    // RocksDB checks for a logger before calling; a direct caller may not.
    return;
  }
  AutoHyperClockCache::ReportProblems(info_log);
  const FIFOCacheStats stats = GetStats();
  ROCKS_LOG_INFO(
      info_log,
      "FIFOCache stats: probation usage %zu, pinned allowance %zu, "
      "promotions %" PRIu64 ", ghost returns %" PRIu64
      ", probation evictions %" PRIu64 ", resident evictions %" PRIu64
      ", effort cap stops %" PRIu64 ", probation full %" PRIu64
      ", pinned twice %" PRIu64 ", resident all pinned %" PRIu64,
      stats.probation_usage, stats.pinned_allowance, stats.promotions,
      stats.ghost_returns, stats.probation_evictions, stats.resident_evictions,
      stats.effort_cap_stops, stats.probation_full, stats.pinned_twice,
      stats.resident_all_pinned);
}

const void* FIFOCache::GetOptionsPtr(const std::string& name) const {
  if (name == kSelfName) {
    return this;
  }
  return AutoHyperClockCache::GetOptionsPtr(name);
}

std::vector<const void*> FIFOCache::TEST_GetMappings() const {
  std::vector<const void*> mappings;
  ForEachShard([&mappings](const Shard* shard) {
    const std::vector<const void*> table = shard->GetTable().TEST_GetMappings();
    mappings.insert(mappings.end(), table.begin(), table.end());
  });
  return mappings;
}

}  // namespace fifo_cache

const std::unordered_map<std::string, OptionTypeInfo>&
FIFOCacheOptionsTypeInfo() {
  static const std::unordered_map<std::string, CacheMetadataChargePolicy>
      metadata_charge_policy_string_map = {
          {"kDontChargeCacheMetadata", kDontChargeCacheMetadata},
          {"kFullChargeCacheMetadata", kFullChargeCacheMetadata},
      };
  static const std::unordered_map<std::string, OptionTypeInfo> type_info = {
      {"capacity",
       {offsetof(struct FIFOCacheOptions, capacity), OptionType::kSizeT,
        OptionVerificationType::kNormal}},
      {"num_shard_bits",
       {offsetof(struct FIFOCacheOptions, num_shard_bits), OptionType::kInt,
        OptionVerificationType::kNormal}},
      {"strict_capacity_limit",
       {offsetof(struct FIFOCacheOptions, strict_capacity_limit),
        OptionType::kBoolean, OptionVerificationType::kNormal}},
      {"metadata_charge_policy",
       OptionTypeInfo::Enum<CacheMetadataChargePolicy>(
           offsetof(struct FIFOCacheOptions, metadata_charge_policy),
           &metadata_charge_policy_string_map)},
      {"hash_seed",
       {offsetof(struct FIFOCacheOptions, hash_seed), OptionType::kInt32T,
        OptionVerificationType::kNormal}},
      {"use_adaptive_mutex",
       {offsetof(struct FIFOCacheOptions, use_adaptive_mutex),
        OptionType::kBoolean, OptionVerificationType::kNormal}},
      {"max_capacity",
       {offsetof(struct FIFOCacheOptions, max_capacity), OptionType::kSizeT,
        OptionVerificationType::kNormal}},
  };
  return type_info;
}

bool GetFIFOCacheStats(const Cache& cache, FIFOCacheStats* stats) {
  const auto* fifo =
      static_cast<const FIFOCache*>(cache.GetOptionsPtr(FIFOCache::kSelfName));
  if (fifo == nullptr) {
    return false;
  }
  *stats = fifo->GetStats();
  return true;
}

namespace {
// CacheWithSecondaryAdapter reports its target's Name(), so a FIFOCache
// wrapped in it looks like a bare one. This adapter also forwards
// FIFOCache's GetOptionsPtr entry, so GetFIFOCacheStats finds the FIFOCache
// behind it.
class FIFOCacheSecondaryAdapter : public CacheWithSecondaryAdapter {
 public:
  using CacheWithSecondaryAdapter::CacheWithSecondaryAdapter;
  const void* GetOptionsPtr(const std::string& name) const override {
    if (name == FIFOCache::kSelfName) {
      return target_->GetOptionsPtr(name);
    }
    return CacheWithSecondaryAdapter::GetOptionsPtr(name);
  }
};
}  // namespace

std::shared_ptr<Cache> FIFOCacheOptions::MakeSharedCache() const {
  if (num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  std::shared_ptr<Cache> cache = std::make_shared<FIFOCache>(*this);
  if (secondary_cache) {
    cache = std::make_shared<FIFOCacheSecondaryAdapter>(cache, secondary_cache);
  }
  return cache;
}

}  // namespace ROCKSDB_NAMESPACE
