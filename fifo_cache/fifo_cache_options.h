//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// FIFOCache's public API. Installed with RocksDB as
// rocksdb/plugin/fifo_cache/fifo_cache_options.h.

#pragma once

#include <cstddef>
#include <cstdint>
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
// resident queue instead of being evicted. The share grows, up to five
// times that, while entries reach the head still held through the handle
// their insert returned, until most are released by the time they get
// there. The resident queue is HyperClockCache's clock, which keeps up to
// three hits per entry and visits entries in the table's order rather than
// the order they arrived. Keys (not values) recently evicted from probation
// are remembered, and inserting one again puts it straight into the
// resident queue. So does a cache reservation, an entry with no value and a
// helper other than kNoopCacheItemHelper that has no deleter, inserted with
// a handle as RocksDB inserts them, and its key is never remembered either;
// such an entry inserted without a handle goes through probation. Placement
// comes from hits alone: Cache::Priority is ignored. GetFIFOCacheStats
// (below) reports what the queues did, and Cache::ReportProblems, which
// RocksDB calls at each stats dump, logs the same at INFO level.
//
// A negative num_shard_bits (the default) gives each shard at least 4 MiB, up
// to 64 shards, so a cache under 8 MiB has one. LRUCache defaults to 512 KiB
// shards, but small shards cost hits and, with no shard mutex, the extra shards
// don't buy throughput: over 15 db_bench workloads with caches of 1% to 30% of
// the data, 4 MiB shards missed 14% less than 512 KiB ones on average (41% less
// at most), and cache_bench at 16 and 64 MiB with 16 and 64 threads ran at
// -3.7% to +5.8% of 512 KiB shards' median ops/sec, inside the run-to-run
// spread. HyperClockCache's 32 MiB minimum missed 0.8% less than 4 MiB on
// average, nearly all of it where a 9 MiB cache had one shard instead of two,
// but one shard of 16 MiB did 4% fewer ops/sec than four at 64 threads. The
// count is set when the cache is built, from capacity and not max_capacity, and
// SetCapacity keeps it, so for a cache that will run well above its starting
// capacity, set num_shard_bits for the capacity it will mostly run at.
//
// The table is HyperClockCache's AutoHyperClockTable, and there is no shard
// mutex: a lookup takes no lock, an insert links its entry without one, and
// erasing or evicting an entry (for an insert too), or growing the table,
// locks only the hash chains it rewrites.
// Inserting a key the cache has replaces the old entry, and Insert returns
// OkOverwritten; a pinned old entry stays, though lookups no longer find
// it, until its last reference is released. (HyperClockCache keeps the old
// entry instead and drops the new value.) The new entry keeps the old one's
// place: it goes to the resident queue if the old one was there, and
// otherwise wherever a new entry would go, and it starts with the old one's
// hits plus one, up to 3, counting the insert as a hit. Of two inserts of
// one key that race, the one linked last stays. One that can't tell which
// that is, because a table split is under way or more than 64 entries are
// linked ahead of both, erases every entry with the key, its own included,
// so the key is missing until it is inserted again, never left with the
// older value.
//
// Pinned entries are never evicted. A probation entry still pinned when
// the queue's head reaches it again, a lap after it went to the tail for
// being pinned, goes to the resident queue, whose clock passes over it
// while it stays pinned, as HyperClockCache's does. A lap of the clock that
// finds nothing in the resident queue but pinned entries (and entries it
// evicts) sends eviction to probation's head first for as many evictions
// as the shard has entries, so a resident entry released meanwhile stays
// until the clock looks again. Without
// strict_capacity_limit, an insert's eviction stops stepping a queue once
// those steps have passed 30 pinned entries for each entry it freed, plus
// 30 (the clock counting the pinned probation entries it passes), and
// gives up when it has stopped both, taking usage over capacity, as in
// HyperClockCache; later inserts evict back as eviction reaches unpinned
// entries. Releasing the last reference to an entry doesn't erase it, even
// over capacity, unless erase_if_last_ref is set. With
// strict_capacity_limit, an insert that cannot make room fails with
// MemoryLimit, or without a handle is dropped and returns OK, as if
// inserted and evicted at once; either way an entry already cached under
// the key stays. A strict insert that charges usage, or finds it full,
// after a SetCapacity goes by the new capacity; one that charged it, or
// found it full, before can still land after the cut, over the new
// capacity until the next insert evicts. A strict insert into a shard over
// capacity evicts the excess with its own charge, and inserts racing it may
// each do so, taking usage below capacity once. Under either limit, an
// insert whose charge would take usage to half of size_t or past it is
// refused the same way, racing inserts included, where HyperClockCache
// lets one pass, and one whose key can't be copied (see below) fails with
// MemoryLimit.
//
// Remembered keys take a 4-byte cell per slot of the table, holding 16 bits
// of the key's hash and the count of probation evictions when it was
// evicted. A key's cell can be any of the eight in the group of slots
// holding the slot it maps to, and counts for as many probation evictions
// as the shard has entries less probation's 12% (rounded down), the size of
// S3-FIFO's ghost, rounded up to the count's step: one eviction for a shard
// of up to about 2.6 MB capacity, 2^n for one of 2^n times that, whatever
// max_capacity (see below) reserves. A SetCapacity that changes the step
// keeps the remembered keys, with their ages in the new step rounded down,
// and forgets the ones too old to count in it. A cell counts for at most 32,768
// steps, which falls short of that share only for entries averaging under 70
// bytes of charge, under kDontChargeCacheMetadata with max_capacity above
// capacity. A victim whose group has no empty cell replaces the oldest, unless
// all eight are from its own step, and while a group's eight cells all count,
// an insert of a key that isn't remembered matches one of them 1 time in 8,192.
// A cell that stopped counting counts again if the shard later holds enough
// entries for that count to cover the evictions since its key's, or 65,280
// steps on. While a SetCapacity converts the stamps to a new step, inserts
// find no remembered keys and remember none, and a key evicted as the
// conversion begins can keep its age in the old step.
// Under kFullChargeCacheMetadata (the default) the table counts against
// capacity: 72 bytes per slot the table has grown to (the 64-byte slot, its
// probation word and its ghost cell), starting at 64 slots per shard, or 4 for
// a shard of 4 KiB or less, and growing with the entries; 8 bytes per cell of
// the probation queue, which adds a larger ring as the table grows and stops
// counting an older one, returning its whole pages to the OS on Linux, once
// evictions have taken all it held (a pop still finishing as it goes can
// map one page back, uncharged); and for a key that isn't 16 bytes long,
// 24 bytes plus the key, which is copied to the heap. The table never
// shrinks. With strict_capacity_limit, the table grows only if the slot
// fits within capacity beside the entry being inserted, and the probation queue
// grows only if it fits within capacity, so after a capacity cut below what the
// table already takes, the shard holds no entries but those whose inserts
// charged usage before the cut.
//
// Each shard reserves address space for max(2048, shard capacity / 64)
// slots up front, using max_capacity instead of capacity when it is larger,
// and touches it as the table grows. On Linux the reservation does not
// count against the commit limit unless vm.overcommit_memory=2; on Windows
// it is committed up front. If it cannot be reserved, construction aborts,
// as for HyperClockCache. The table cannot grow past its reservation, which
// holds about one entry per 80 bytes of reserved capacity: once
// SetCapacity raises capacity beyond that (about 51x for 4 KiB entries,
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

// A FIFOCache's S3-FIFO state, summed over its shards: probation's usage
// now, and counts since the cache was created. RocksDB reports the rest for
// any cache: hits, misses and adds per block role in the BLOCK_CACHE_*
// tickers, usage, pinned usage, capacity and entries per role in the
// rocksdb.block-cache-* properties, and Cache::GetOccupancyCount.
struct FIFOCacheStats {
  // Charge of the entries in the probation queues.
  size_t probation_usage = 0;
  // How far probation's share is above 12% of capacity now, for entries
  // still held through their insert's handle when they reached the head.
  size_t pinned_allowance = 0;
  // Entries promoted from the head of a probation queue to the resident
  // queue, having been hit at least twice.
  uint64_t promotions = 0;
  // Inserts of a key remembered from a probation eviction, whose entries
  // went straight into the resident queue. Against probation_evictions it
  // shows how many of probation's victims came back while remembered
  // (the placeholders CacheWithSecondaryAdapter inserts count among those
  // victims, but are never remembered).
  uint64_t ghost_returns = 0;
  // Entries eviction removed from the probation queues and from the
  // resident queue, for an insert or a capacity cut. Entries erased,
  // replaced or removed by EraseUnRefEntries are not counted, except for
  // one erased while pinned that eviction reaches as its last reference is
  // released, before the release frees it.
  uint64_t probation_evictions = 0;
  uint64_t resident_evictions = 0;
  // Times eviction gave up at the effort cap (see above) having freed less
  // than it needed: once per insert whose eviction did, after which the
  // insert takes usage over capacity (or, if it needed a slot the table
  // didn't have, goes standalone or is dropped), and once per capacity cut
  // that ends over the new capacity having given up at the cap. Never under
  // strict_capacity_limit.
  uint64_t effort_cap_stops = 0;
  // Entries that went to the resident queue, on insert or on being moved
  // to the tail of probation (as a pinned head is), because the probation
  // queue had no room for them. A push into a full queue takes up to eight
  // cells from the oldest end of its newest ring, putting back those of
  // entries still queued, and takes the place of the first one left by an
  // entry erased while queued. If there is none, its entry goes to the
  // resident queue, and so does an entry it took out whose room another
  // push took. That happens in a shard's first ring as the shard first
  // fills while entries are erased (an insert replacing an entry under its
  // key erases it, and a secondary cache's adapter erases the placeholder it
  // inserts on a block's first hit there at the next lookup). A push also
  // sends its entry there when the table has outgrown the newest ring and a
  // new one couldn't be added: under strict_capacity_limit, when it didn't
  // fit within capacity.
  uint64_t probation_full = 0;
  // Entries that went to the resident queue because a probation step found
  // them pinned a lap after it had moved them to the tail for being pinned:
  // entries held for longer than the queue's laps, as a table reader holds
  // pinned index and filter blocks, or that lookups keep pinned. The
  // resident clock passes over them while they stay pinned, as
  // HyperClockCache's does. Each one its inserter alone held on its first
  // visit added its charge to pinned_allowance then.
  uint64_t pinned_twice = 0;
  // Laps of the resident clock that evicted nothing and took no hit off
  // any entry, having found nothing but pinned entries, less those a step
  // still under way when the lap ended found something in while the rest
  // the lap set still stood, and those whose rest would have been over as
  // it was set (the shard had no entries and the call nothing pending, or
  // other evictions took the count past its end meanwhile). After each,
  // eviction takes probation's head first, from the rest of that eviction
  // on, for as many probation evictions as the shard has entries (up to
  // 2^29), unless a step of the clock (from an eviction already under way,
  // or one whose probation has nothing to pop) finds something to evict or
  // a hit to take off first, so a resident entry released meanwhile mostly
  // stays until then.
  uint64_t resident_all_pinned = 0;
};

// Sets *stats to the FIFOCacheStats of the FIFOCache that cache is, or that
// MakeSharedCache wrapped for a secondary cache. Other threads keep
// counting while the shards are read, so the counts need not agree with
// each other at one instant. Returns false and leaves *stats unchanged for
// any other cache, a FIFOCache wrapped some other way included.
bool GetFIFOCacheStats(const Cache& cache, FIFOCacheStats* stats);

}  // namespace ROCKSDB_NAMESPACE
