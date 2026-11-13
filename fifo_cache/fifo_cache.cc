//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "plugin/fifo_cache/fifo_cache.h"

#include <cstddef>
#include <memory>

#include "cache/secondary_cache_adapter.h"

namespace ROCKSDB_NAMESPACE {
namespace fifo_cache {

namespace {
// The table's options. The slice assignment sets every ShardedCacheOptions
// field, capacity included; estimated_entry_charge keeps its default of 0,
// which AutoHyperClockTable requires. use_adaptive_mutex and max_capacity
// have nothing to configure here. num_shard_bits left negative takes
// RocksDB's default for the capacity.
HyperClockCacheOptions TableOptions(const FIFOCacheOptions& opts) {
  HyperClockCacheOptions table_opts(/*capacity*/ 0);
  static_cast<ShardedCacheOptions&>(table_opts) = opts;
  if (table_opts.num_shard_bits < 0) {
    table_opts.num_shard_bits = GetDefaultCacheShardBits(table_opts.capacity);
  }
  return table_opts;
}
}  // namespace

FIFOCache::FIFOCache(const FIFOCacheOptions& opts)
    : AutoHyperClockCache(TableOptions(opts)) {}

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

std::shared_ptr<Cache> FIFOCacheOptions::MakeSharedCache() const {
  if (num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  std::shared_ptr<Cache> cache = std::make_shared<FIFOCache>(*this);
  if (secondary_cache) {
    cache = std::make_shared<CacheWithSecondaryAdapter>(cache, secondary_cache);
  }
  return cache;
}

}  // namespace ROCKSDB_NAMESPACE
