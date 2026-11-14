//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// FIFOCache's public API. Installed with RocksDB as
// rocksdb/plugin/fifo_cache/fifo_cache_options.h.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// EXPERIMENTAL: the API and the fifo_cache:// option string are subject to
// change.
// FIFOCache - A cache using S3-FIFO eviction to stay at or below a set
// capacity. The cache is sharded to 2^num_shard_bits shards, by hash of the
// key, and the total capacity is divided evenly among the shards. Each shard
// is a hash table with two queues over its entries: new entries enter the
// probation queue, a FIFO whose share is 12% of capacity, and an entry hit
// at least twice by the time it reaches the head is promoted to the
// resident queue instead of being evicted. The resident queue is a clock
// over the table that keeps up to three hits per entry. Keys (not values)
// recently evicted from probation are remembered, and inserting one again
// puts it straight into the resident queue. Placement comes from hits
// alone: Cache::Priority is ignored.
//
// The table is HyperClockCache's AutoHyperClockTable, and there is no shard
// mutex: a lookup takes no lock, an insert links its entry without one, and
// erasing or evicting an entry (for an insert too), or growing the table,
// locks only the hash chains it rewrites.
// Inserting a key the cache has replaces the old entry, and Insert returns
// OkOverwritten; a pinned old entry stays, though lookups no longer find
// it, until its last reference is released. (HyperClockCache keeps the old
// entry instead and drops the new value.)
//
// Pinned entries are never evicted. Without strict_capacity_limit, an
// insert's eviction gives up once it has passed 30 pinned entries for each
// entry it freed, plus 30, and the insert takes usage over capacity, as in
// HyperClockCache; later inserts evict back as eviction reaches unpinned
// entries. Releasing the last reference to an entry doesn't erase it, even
// over capacity, unless erase_if_last_ref is set. With
// strict_capacity_limit, an insert that cannot make room fails with
// MemoryLimit, or without a handle is dropped and returns OK, as if
// inserted and evicted at once; either way an entry already cached under
// the key stays. Under either limit, an insert whose charge would take
// usage past half of size_t is refused the same way, and one whose key
// can't be copied (see below) fails with MemoryLimit.
//
// Remembered keys take a 4-byte tag per slot of the table, at the slot the
// key maps to, so a later victim mapping to the same slot replaces the tag.
// Under kFullChargeCacheMetadata (the default) the table counts against
// capacity: 72 bytes per slot the table has grown to (the 64-byte slot, its
// probation word and its tag), starting at 64 slots per shard, or 4 for a
// shard of 4 KiB or less, and growing with the entries; 16 bytes per cell
// of the probation queue, which grows with the table; and for a key that
// isn't 16 bytes long, 24 bytes plus the key, which is copied to the heap.
// The table never shrinks. With strict_capacity_limit, the table grows only
// if the slot fits within capacity beside the entry being inserted, and the
// probation queue grows only if it fits within capacity, so after a
// capacity cut below what the table already takes, the shard holds no
// entries.
//
// Each shard reserves address space for max(2048, shard capacity / 64)
// slots up front, using max_capacity instead of capacity when it is larger,
// and touches it as the table grows. On Linux the reservation does not
// count against the commit limit unless vm.overcommit_memory=2; on Windows
// it is committed up front. If it cannot be reserved, construction aborts,
// as for HyperClockCache. The table cannot grow past its reservation, which
// holds about one entry per 107 bytes of reserved capacity: once
// SetCapacity raises capacity beyond that (about 38x for 4 KiB entries,
// with the default max_capacity), new keys are also evicted by occupancy.
//
// This is an opt-in implementation and not the default block cache. Besides
// MakeSharedCache, a RocksDB built with this plugin creates one from
// "fifo_cache://capacity=<n>;..." with the scalar fields below, as the whole
// value given to Cache::CreateFromString with an empty result (e.g. db_bench
// --cache_uri). As the id of a block_cache option value, in an option
// string or OPTIONS file, the URI carries a single option (e.g.
// block_cache={id=fifo_cache://capacity=1G}). memory_allocator and
// secondary_cache can only be set in C++.
struct FIFOCacheOptions : public ShardedCacheOptions {
  // Ignored, as the cache has no mutex to configure. It is still accepted,
  // true or false, so options that set it keep parsing.
  bool use_adaptive_mutex = kDefaultToAdaptiveMutex;

  // Capacity the slot table is reserved for, when larger than capacity (see
  // above). Set it to the largest capacity SetCapacity may raise this cache
  // to; 0 (the default) reserves for capacity. On Linux a larger
  // reservation costs only address space until the table grows into it; on
  // Windows it is committed up front.
  size_t max_capacity = 0;

  FIFOCacheOptions() {}
  // Every argument is required, so a positional call written for another
  // cache's options does not compile.
  FIFOCacheOptions(size_t _capacity, int _num_shard_bits,
                   bool _strict_capacity_limit,
                   std::shared_ptr<MemoryAllocator> _memory_allocator,
                   bool _use_adaptive_mutex,
                   CacheMetadataChargePolicy _metadata_charge_policy)
      : ShardedCacheOptions(_capacity, _num_shard_bits, _strict_capacity_limit,
                            std::move(_memory_allocator),
                            _metadata_charge_policy),
        use_adaptive_mutex(_use_adaptive_mutex) {}

  // Construct an instance of FIFOCache using these options
  std::shared_ptr<Cache> MakeSharedCache() const;
};

}  // namespace ROCKSDB_NAMESPACE
