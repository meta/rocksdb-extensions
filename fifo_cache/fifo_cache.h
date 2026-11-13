//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
#pragma once

#include <string>
#include <unordered_map>

#include "plugin/fifo_cache/clock_table.h"
#include "plugin/fifo_cache/fifo_cache_options.h"
#include "rocksdb/utilities/options_type.h"

namespace ROCKSDB_NAMESPACE {
namespace fifo_cache {

// FIFOCache keeps its entries in AutoHyperClockCache's slot table, copied
// into clock_table.h. There is no shard mutex: lookups, releases that don't
// erase, and inserts that neither evict nor grow the table are wait-free,
// and removing an entry from a chain or splitting a chain to grow the table
// takes a lock on that chain only.
class FIFOCache
#ifdef NDEBUG
    final
#endif
    : public AutoHyperClockCache {
 public:
  explicit FIFOCache(const FIFOCacheOptions& opts);
  const char* Name() const override { return "FIFOCache"; }
};

}  // namespace fifo_cache

using FIFOCache = fifo_cache::FIFOCache;

// The fifo_cache:// option fields. Built on first use, so it is ready even
// when called during another translation unit's static initialization.
const std::unordered_map<std::string, OptionTypeInfo>&
FIFOCacheOptionsTypeInfo();

}  // namespace ROCKSDB_NAMESPACE
