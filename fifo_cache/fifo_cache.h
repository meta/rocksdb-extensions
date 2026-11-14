//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plugin/fifo_cache/clock_table.h"
#include "plugin/fifo_cache/fifo_cache_options.h"
#include "rocksdb/utilities/options_type.h"

namespace ROCKSDB_NAMESPACE {
namespace fifo_cache {

// FIFOCache keeps its entries in AutoHyperClockCache's slot table, copied
// into clock_table.h. There is no shard mutex: lookups, releases that don't
// erase and inserts take no lock, and removing an entry from a chain
// (erasing or evicting it, or an insert replacing it) or splitting a chain
// to grow the table takes a lock on that chain only.
class FIFOCache
#ifdef NDEBUG
    final
#endif
    : public AutoHyperClockCache {
 public:
  explicit FIFOCache(const FIFOCacheOptions& opts);
  const char* Name() const override { return "FIFOCache"; }

  // The shards' FIFOCacheStats summed, as GetFIFOCacheStats returns them.
  FIFOCacheStats GetStats() const;

  // What HyperClockCache logs at DEBUG level, then GetStats at INFO level.
  void ReportProblems(const std::shared_ptr<Logger>& info_log) const override;

  // GetOptionsPtr(kSelfName) returns this FIFOCache. The adapter
  // MakeSharedCache wraps it in for a secondary cache reports the same
  // Name() and forwards this one name, so GetFIFOCacheStats finds the
  // FIFOCache behind either without RTTI and never casts anything else.
  static constexpr const char* kSelfName = "fifo_cache::FIFOCache";
  const void* GetOptionsPtr(const std::string& name) const override;

  // Total charge of the entries in the probation queues, and of those last
  // seen pinned, which don't count toward probation's share.
  size_t TEST_GetProbationUsage() const;
  size_t TEST_GetProbationPinnedUsage() const;
  // Probation's share of capacity, summed over shards.
  size_t TEST_GetProbationTarget() const;
  // What kFullChargeCacheMetadata charges for the tables: slots, their
  // probation words and ghost tags, and the probation queues' rings.
  size_t TEST_GetMetadataCharge() const;
  // Whether the key is cached in a probation queue. Takes no hit.
  bool TEST_InProbation(const Slice& key);
  // Whether an insert of the key would find it in the ghost.
  bool TEST_InGhost(const Slice& key) const;
  // The start of each shard's table mappings (see
  // AutoHyperClockTable::TEST_GetMappings).
  std::vector<const void*> TEST_GetMappings() const;
};

}  // namespace fifo_cache

using FIFOCache = fifo_cache::FIFOCache;

// The fifo_cache:// option fields. Built on first use, so it is ready even
// when called during another translation unit's static initialization.
const std::unordered_map<std::string, OptionTypeInfo>&
FIFOCacheOptionsTypeInfo();

}  // namespace ROCKSDB_NAMESPACE
