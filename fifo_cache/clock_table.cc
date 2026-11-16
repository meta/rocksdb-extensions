//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// FIFOCache's slot table: a copy of RocksDB's cache/clock_cache.cc from
// v11.8.1 (abeebd9630f11bd08c28b7bd43c7bdfc62050654). The copy is in
// namespace fifo_cache, leaves out FixedHyperClockTable,
// FixedHyperClockCache, NewClockCache and
// HyperClockCacheOptions::MakeSharedCache, lets FIFOCache derive from
// AutoHyperClockCache, and names FIFOCache in its messages. TryInsert is
// gone, replaced by AutoHyperClockTable::TryInsertSlot. Every other change
// to the copied code has a comment starting with "FIFOCache:".
// Diff against that revision to see them all, and to carry over later
// upstream fixes.

#include "plugin/fifo_cache/clock_table.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>

#ifdef OS_LINUX
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "cache/cache_key.h"
#include "cache/secondary_cache_adapter.h"
#include "logging/logging.h"
#include "plugin/fifo_cache/fifo_cache_options.h"
#include "port/likely.h"
#include "rocksdb/env.h"
#include "test_util/sync_point.h"  // FIFOCache: the probation sync points
#include "util/autovector.h"
#include "util/hash.h"
#include "util/math.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace fifo_cache {

namespace {
using SlotMeta = ClockHandle::SlotMeta;
using AcquireCounter = SlotMeta::AcquireCounter;
using ReleaseCounter = SlotMeta::ReleaseCounter;

// FIFOCache: Cache::Priority is ignored, and counters start at zero hits:
// the one count is the reference DoInsert holds while linking the entry,
// which it drops, or hands to the caller with kCountsInsertRef set.
inline uint32_t GetInitialCountdown(Cache::Priority /*priority*/) { return 1; }

inline void MarkEmpty(ClockHandle& h) {
#ifndef NDEBUG
  // Mark slot as empty, with assertion
  auto old_meta = h.meta.Exchange({});
  assert(old_meta.IsUnderConstruction());
#else
  // Mark slot as empty
  h.meta.Store({});
#endif
}

// Called to undo the effect of referencing an entry for internal purposes,
// so it should not be marked as having been used.
inline void Unref(const ClockHandle& h, uint32_t count = 1) {
  // Pretend we never took the reference
  // FIFOCache: only where the caller can't free the entry, in scans and on a
  // mismatch in MatchAndRef. An erased entry whose last reference is dropped
  // here is freed by the next lookup that takes and drops a reference to
  // it, or by eviction.
  SlotMeta old_meta;
  h.meta.Apply(AcquireCounter::MinusTransformPromiseNoUnderflow(count),
               &old_meta);
  assert(old_meta.GetRefcount() != 0);
  (void)old_meta;
}

// FIFOCache: the resident queue's clock, for entries not in the probation
// queue (PurgeImplLocked skips those).
inline bool ClockUpdate(ClockHandle& h, BaseClockTable::EvictionData* data,
                        bool* purgeable = nullptr) {
  SlotMeta meta;
  if (purgeable) {
    assert(*purgeable == false);
    // In AutoHCC, our eviction process follows the chain structure, so we
    // should ensure that we see the latest state of each entry, at least for
    // assertion checking.
    meta = h.meta.Load();
  } else {
    // In FixedHCC, our eviction process is a simple iteration without regard
    // to probing order, displacements, etc., so it doesn't matter if we see
    // somewhat stale data.
    meta = h.meta.LoadRelaxed();
  }

  if (!meta.IsShareable()) {
    // Only clock update Shareable entries
    if (purgeable) {
      *purgeable = true;
      // AutoHCC only: make sure we only attempt to update non-empty slots
      assert(!meta.IsEmpty());
    }
    return false;
  }
  uint32_t acquire_count = meta.GetAcquireCounter();
  uint32_t release_count = meta.GetReleaseCounter();
  if (acquire_count != release_count) {
    // Only clock update entries with no outstanding refs
    data->seen_pinned_count++;
    return false;
  }
  if (meta.IsVisible() && acquire_count > 0) {
    // FIFOCache: an entry the clock can take once its countdown runs out.
    data->clock_countdowns++;
    // Decrement clock
    uint32_t new_count =
        std::min(acquire_count - 1, uint32_t{ClockHandle::kMaxCountdown} - 1);
    // Compare-exchange in the decremented clock info, but
    // not aggressively
    SlotMeta new_meta = meta;
    new_meta.SetReleaseCounter(new_count);
    new_meta.SetAcquireCounter(new_count);
    h.meta.CasStrongRelaxed(meta, new_meta);
    return false;
  }
  // Otherwise, remove entry (either unreferenced invisible or
  // unreferenced and expired visible).
  SlotMeta construction_meta;
  construction_meta.SetUnderConstruction();
  construction_meta.SetHit(meta.GetHit());
  if (h.meta.CasStrong(meta, construction_meta)) {
    // Took ownership.
    data->freed_charge += h.GetTotalCharge();
    data->freed_count += 1;
    return true;
  } else {
    // Compare-exchange failing probably
    // indicates the entry was used, so skip it in that case.
    return false;
  }
}

// If an entry doesn't receive clock updates but is repeatedly referenced &
// released, the acquire and release counters could overflow without some
// intervention. This is that intervention, which should be inexpensive
// because it only incurs a simple, very predictable check. (Applying a bit
// mask in addition to an increment to every Release likely would be
// relatively expensive, because it's an extra atomic update.)
//
// We do have to assume that we never have many millions of simultaneous
// references to a cache handle, because we cannot represent so many
// references with the difference in counters, masked to the number of
// counter bits. Similarly, we assume there aren't millions of threads
// holding transient references (which might be "undone" rather than
// released by the way).
//
// Consider these possible states for each counter:
// low: less than kMaxCountdown
// medium: kMaxCountdown to half way to overflow + kMaxCountdown
// high: half way to overflow + kMaxCountdown, or greater
//
// And these possible states for the combination of counters:
// acquire / release
// -------   -------
// low       low       - Normal / common, with caveats (see below)
// medium    low       - Can happen while holding some refs
// high      low       - Violates assumptions (too many refs)
// low       medium    - Violates assumptions (refs underflow, etc.)
// medium    medium    - Normal (very read heavy cache)
// high      medium    - Can happen while holding some refs
// low       high      - This function is supposed to prevent
// medium    high      - Violates assumptions (refs underflow, etc.)
// high      high      - Needs CorrectNearOverflow
//
// Basically, this function detects (high, high) state (inferred from
// release alone being high) and bumps it back down to (medium, medium)
// state with the same refcount and the same logical countdown counter
// (everything > kMaxCountdown is logically the same). Note that bumping
// down to (low, low) would modify the countdown counter, so is "reserved"
// in a sense.
//
// If near-overflow correction is triggered here, there's no guarantee
// that another thread hasn't freed the entry and replaced it with another.
// Therefore, it must be the case that the correction does not affect
// entries unless they are very old (many millions of acquire-release cycles).
// (Our bit manipulation is indeed idempotent and only affects entries in
// exceptional cases.) We assume a pre-empted thread will not stall that long.
// If it did, the state could be corrupted in the (unlikely) case that the top
// bit of the acquire counter is set but not the release counter, and thus
// we only clear the top bit of the acquire counter on resumption. It would
// then appear that there are too many refs and the entry would be permanently
// pinned (which is not terrible for an exceptionally rare occurrence), unless
// it is referenced enough (at least kMaxCountdown more times) for the release
// counter to reach "high" state again and bumped back to "medium." (This
// motivates only checking for release counter in high state, not both in high
// state.)
inline void CorrectNearOverflow(SlotMeta old_meta,
                                BitFieldsAtomic<SlotMeta>& meta) {
  // We clear both top-most counter bits at the same time.
  constexpr uint32_t kCounterTopBit = uint32_t{1}
                                      << (SlotMeta::kCounterNumBits - 1);
  // The threshold for correcting "near overflow" is to ensure
  // (a) the value has a top bit set that can be cleared
  // (b) when we clear the top bit, the eviction state will be preserved
  //     (everything >= kMaxCountdown is treated equivalently)
  // As mentioned above, we only check the release count.
  constexpr uint32_t kThreshold = kCounterTopBit + ClockHandle::kMaxCountdown;

  if (UNLIKELY(old_meta.GetReleaseCounter() > kThreshold)) {
    auto clear_transform = AcquireCounter::AndTransform(kCounterTopBit - 1) +
                           ReleaseCounter::AndTransform(kCounterTopBit - 1);
    meta.ApplyRelaxed(clear_transform);
  }
}

// FIFOCache: an occupied slot isn't checked for the key. The insert links
// its entry and then erases any other with the key (EraseReplaced), where
// HyperClockCache keeps a visible match it finds here, counting a hit on it,
// and drops the new value.
inline bool BeginSlotInsert(ClockHandle& h) {
  // Optimistically transition the slot from "empty" to
  // "under construction" (no effect on other states)
  auto set_occupied = SlotMeta::OccupiedFlag::SetTransform();
  SlotMeta old_meta;
  h.meta.Apply(set_occupied, &old_meta);
  // Whether we've started inserting into an available slot, and taken
  // ownership.
  return old_meta.IsEmpty();
}

inline void FinishSlotInsert(const ClockHandleBasicData& proto, ClockHandle& h,
                             uint32_t initial_countdown, bool keep_ref) {
  // Save data fields
  ClockHandleBasicData* h_alias = &h;
  *h_alias = proto;

  // Transition from "under construction" state to "visible" state
  SlotMeta new_meta;
  new_meta.SetVisible();

  // Maybe with an outstanding reference
  new_meta.SetAcquireCounter(initial_countdown);
  new_meta.SetReleaseCounter(initial_countdown - (keep_ref ? 1 : 0));

#ifndef NDEBUG
  // Save the state transition, with assertion
  auto old_meta = h.meta.Exchange(new_meta);
  assert(old_meta.IsUnderConstruction());
#else
  // Save the state transition
  h.meta.Store(new_meta);
#endif
}

// Func must be const HandleImpl& -> void callable
template <class HandleImpl, class Func>
void ConstApplyToEntriesRange(const Func& func, const HandleImpl* begin,
                              const HandleImpl* end,
                              bool apply_if_will_be_deleted) {
  for (const HandleImpl* h = begin; h < end; ++h) {
    // Note: to avoid using compare_exchange, we have to be extra careful.
    SlotMeta old_meta = h->meta.LoadRelaxed();
    // Check if it's an entry visible to lookups
    if (apply_if_will_be_deleted || old_meta.IsVisible()) {
      if (old_meta.IsShareable()) {
        // Increment acquire counter. Note: it's possible that the entry has
        // completely changed since we loaded old_meta, but incrementing acquire
        // count is always safe. (Similar to optimistic Lookup here.)
        auto add_acquire = AcquireCounter::PlusTransformPromiseNoOverflow(1);
        h->meta.Apply(add_acquire, &old_meta);
        // Check whether we actually acquired a reference.
        if (old_meta.IsShareable()) {
          // Apply func if appropriate
          if (apply_if_will_be_deleted || old_meta.IsVisible()) {
            func(*h);
          }
          // Pretend we never took the reference
          Unref(*h);
          // No net change, so don't need to check for overflow
        } else {
          // For other states, incrementing the acquire counter has no effect
          // so we don't need to undo it. Furthermore, we cannot safely undo
          // it because we did not acquire a read reference to lock the
          // entry in a Shareable state.
        }
      }
    }
  }
}

uint32_t SanitizeEvictionEffortCap(int eviction_effort_cap) {
  eviction_effort_cap = std::max(int{1}, eviction_effort_cap);
  return static_cast<uint32_t>(eviction_effort_cap);
}

}  // namespace

void ClockHandleBasicData::FreeData(MemoryAllocator* allocator) const {
  const Cache::CacheItemHelper* helper = GetHelper();
  if (helper->del_cb) {
    helper->del_cb(value, allocator);
  }
  FreeHeapKey();
}

#ifndef NDEBUG
namespace {
std::atomic<int64_t> live_heap_keys{0};
}  // namespace

int64_t TEST_LiveHeapKeys() { return live_heap_keys.load(); }
#endif  // !NDEBUG

void ClockHandleBasicData::FreeHeapKey() const {
  if (HasHeapKey()) {
    free(const_cast<HeapKey*>(GetHeapKey()));
#ifndef NDEBUG
    live_heap_keys.fetch_sub(1);
#endif  // !NDEBUG
  }
}

bool ClockHandleBasicData::SetKeyAndHelper(const Slice& key,
                                           const Cache::CacheItemHelper* helper,
                                           CacheMetadataChargePolicy policy) {
  static_assert(alignof(HeapKey) > 1 && alignof(Cache::CacheItemHelper) > 1,
                "the low bit marks a HeapKey");
  if (key.size() == kCacheKeySize) {
    helper_or_heap_key = reinterpret_cast<uintptr_t>(helper);
    return true;
  }
  constexpr size_t kHeader = sizeof(HeapKey);
  if (key.size() > SIZE_MAX - kHeader) {
    return false;
  }
  auto* heap_key = static_cast<HeapKey*>(malloc(kHeader + key.size()));
  if (heap_key == nullptr) {
    return false;
  }
#ifndef NDEBUG
  live_heap_keys.fetch_add(1);
#endif  // !NDEBUG
  heap_key->helper = helper;
  heap_key->size = key.size();
  if (key.size() > 0) {
    memcpy(heap_key->data(), key.data(), key.size());
  }
  heap_key->charge = 0;
  if (policy == kFullChargeCacheMetadata) {
    // Saturating, so a huge charge doesn't wrap.
    const size_t own = kHeader + key.size();
    heap_key->charge = std::min(own, SIZE_MAX - total_charge);
    total_charge += heap_key->charge;
  }
  helper_or_heap_key = reinterpret_cast<uintptr_t>(heap_key) | 1;
  return true;
}

void ClockHandleBasicData::ClearCharge() {
  total_charge = 0;
  if (HasHeapKey()) {
    const_cast<HeapKey*>(GetHeapKey())->charge = 0;
  }
}

Slice ClockHandleBasicData::GetKey(UniqueId64x2* unhashed,
                                   uint32_t seed) const {
  if (HasHeapKey()) {
    return GetHeapKey()->key();
  }
  return ClockCacheShard<AutoHyperClockTable>::ReverseHash(hashed_key, unhashed,
                                                           seed);
}

BaseClockTable::BaseClockTable(size_t capacity, bool strict_capacity_limit,
                               int eviction_effort_cap,
                               CacheMetadataChargePolicy metadata_charge_policy,
                               MemoryAllocator* allocator,
                               const Cache::EvictionCallback* eviction_callback,
                               const uint32_t* hash_seed)
    : capacity_(capacity),
      eec_and_scl_(EecAndScl{}
                       .With<EvictionEffortCap>(
                           SanitizeEvictionEffortCap(eviction_effort_cap))
                       .With<StrictCapacityLimit>(strict_capacity_limit)),
      metadata_charge_policy_(metadata_charge_policy),
      allocator_(allocator),
      eviction_callback_(*eviction_callback),
      hash_seed_(*hash_seed) {}

template <class HandleImpl>
HandleImpl* BaseClockTable::StandaloneInsert(
    const ClockHandleBasicData& proto) {
  // Heap allocated separate from table
  HandleImpl* h = new HandleImpl();
  ClockHandleBasicData* h_alias = h;
  *h_alias = proto;
  h->SetStandalone();
  // Single reference (standalone entries only created if returning a refed
  // Handle back to user)
  SlotMeta meta;
  meta.SetInvisible();
  meta.SetAcquireCounter(1);
  h->meta.Store(meta);
  // Keep track of how much of usage is standalone
  standalone_usage_.FetchAddRelaxed(proto.GetTotalCharge());
  return h;
}

namespace {
// FIFOCache: a probation ring is charged when an insert's push finds the
// newest ring full, in DoInsert or in a rotation during the insert's own
// eviction, so after the insert evicted for its own charge. Without the
// strict limit, the insert (or CreateStandalone) evicts for the ring's bytes
// on the way out, however it ends, leaving anything else over capacity to
// later inserts as before. Under the strict limit a ring is charged only if
// it fits, and this finds nothing to evict.
template <class Table>
class EvictForChargedRing {
 public:
  EvictForChargedRing(Table& table, const typename Table::InsertState& state)
      : table_(table), state_(state) {}
  ~EvictForChargedRing() {
    if (UNLIKELY(state_.ring_charge > 0)) {
      table_.EvictOverCapacity(state_.ring_charge);
    }
  }
  EvictForChargedRing(const EvictForChargedRing&) = delete;
  EvictForChargedRing& operator=(const EvictForChargedRing&) = delete;

 private:
  Table& table_;
  const typename Table::InsertState& state_;
};
}  // namespace

template <class Table>
typename Table::HandleImpl* BaseClockTable::CreateStandalone(
    ClockHandleBasicData& proto, bool allow_uncharged) {
  Table& derived = static_cast<Table&>(*this);
  typename Table::InsertState state;
  derived.StartInsert(state);
  EvictForChargedRing<Table> evict_for_ring(derived, state);

  const size_t total_charge = proto.GetTotalCharge();
  // NOTE: we can use eec_and_scl as eviction_effort_cap below because
  // strict_capacity_limit=true is supposed to disable the limit on eviction
  // effort, and a large value effectively does that.
  if (UNLIKELY(!UsageCanTake(total_charge))) {
    // FIFOCache
    if (!allow_uncharged) {
      return nullptr;
    }
    proto.ClearCharge();
  } else if (eec_and_scl_.LoadRelaxed().Get<StrictCapacityLimit>()) {
    Status s = ChargeUsageMaybeEvictStrict<Table>(
        total_charge,
        /*need_evict_for_occupancy=*/false, state);
    if (!s.ok()) {
      if (allow_uncharged) {
        proto.ClearCharge();
      } else {
        return nullptr;
      }
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict<Table>(
        total_charge,
        /*need_evict_for_occupancy=*/false, state);
    if (!success) {
      // Force the issue
      usage_.FetchAddRelaxed(total_charge);
    }
  }

  return StandaloneInsert<typename Table::HandleImpl>(proto);
}

template <class Table>
Status BaseClockTable::ChargeUsageMaybeEvictStrict(
    size_t total_charge, bool need_evict_for_occupancy,
    typename Table::InsertState& state) {
  size_t capacity;
  size_t old_usage;
  size_t new_usage;
  for (;;) {
    capacity = capacity_.LoadRelaxed();
    TEST_SYNC_POINT("FIFOCache::ChargeUsageMaybeEvictStrict:Loaded");
    if (total_charge > capacity) {
      return Status::MemoryLimit(
          "Cache entry too large for a single cache shard: " +
          std::to_string(total_charge) + " > " + std::to_string(capacity));
    }
    // Grab any available capacity, and free up any more required.
    old_usage = usage_.LoadRelaxed();
    do {
      // FIFOCache: on the usage it adds to, which inserts racing since
      // Insert's check (UsageCanTake) can have raised, so that they don't
      // take usage to half of size_t or wrap it.
      if (UNLIKELY(!UsageCanTake(old_usage, total_charge))) {
        return Status::MemoryLimit("Cache entry too large to account for: " +
                                   std::to_string(total_charge));
      }
      new_usage = std::min(capacity, old_usage + total_charge);
      if (new_usage == old_usage) {
        // No change needed
        break;
      }
      // FIFOCache: release-acquire, as SetCapacity's update of usage_ is.
    } while (!usage_.CasWeak(old_usage, new_usage));
    // FIFOCache: a capacity cut that came before the charge above in usage_'s
    // order of updates is visible here, and the charge goes by it: undone, and
    // made again against the new capacity. A cut that came after it sees the
    // charge in usage (see EvictOverCapacity). An insert with no charge to
    // add (new_usage == old_usage, so usage is at the capacity it loaded),
    // which evicts as much as it inserts, checks the capacity again too, so
    // one that comes after a cut goes by it.
    if (capacity_.LoadRelaxed() >= capacity) {
      break;
    }
    if (new_usage != old_usage) {
      usage_.FetchSubRelaxed(new_usage - old_usage);
    }
  }
  // How much do we need to evict then?
  size_t need_evict_charge = old_usage + total_charge - new_usage;
  // FIFOCache: an eviction for occupancy alone asks Evict for an entry and
  // no charge, where HyperClockCache asks for a charge of 1.
  if (need_evict_charge > 0 || UNLIKELY(need_evict_for_occupancy)) {
    EvictionData data;
    // FIFOCache: Evict adds the entry's probation charge with its own
    // changes to the counters (see InsertState::probation_precharged).
    assert(!state.probation_precharged);
    data.probation_precharge = state.probation_charge;
    static_cast<Table*>(this)->Evict(need_evict_charge,
                                     need_evict_for_occupancy, state, &data);
    state.probation_precharged = data.probation_precharge > 0;
    if (UNLIKELY(data.effort_cap_reached)) {
      eviction_effort_exceeded_count_.FetchAddRelaxed(1);
    }
    occupancy_.FetchSub(data.freed_count);
    if (LIKELY(data.freed_charge > need_evict_charge)) {
      assert(data.freed_count > 0);
      // Evicted more than enough
      usage_.FetchSubRelaxed(data.freed_charge - need_evict_charge);
    } else if (data.freed_charge < need_evict_charge ||
               (UNLIKELY(need_evict_for_occupancy) && data.freed_count == 0)) {
      // Roll back to old usage minus evicted
      usage_.FetchSubRelaxed(data.freed_charge + (new_usage - old_usage));
      if (data.freed_charge < need_evict_charge) {
        return Status::MemoryLimit(
            "Insert failed because unable to evict entries to stay within "
            "capacity limit.");
      } else {
        return Status::MemoryLimit(
            "Insert failed because unable to evict entries to stay within "
            "table occupancy limit.");
      }
    }
    // If we needed to evict something and we are proceeding, we must have
    // evicted something.
    assert(data.freed_count > 0);
  }
  return Status::OK();
}

template <class Table>
inline bool BaseClockTable::ChargeUsageMaybeEvictNonStrict(
    size_t total_charge, bool need_evict_for_occupancy,
    typename Table::InsertState& state) {
  // For simplicity, we consider that either the cache can accept the insert
  // with no evictions, or we must evict enough to make (at least) enough
  // space. It could lead to unnecessary failures or excessive evictions in
  // some extreme cases, but allows a fast, simple protocol. If we allow a
  // race to get us over capacity, then we might never get back to capacity
  // limit if the sizes of entries allow each insertion to evict the minimum
  // charge. Thus, we should evict some extra if it's not a signifcant
  // portion of the shard capacity. This can have the side benefit of
  // involving fewer threads in eviction.
  const size_t old_usage = usage_.LoadRelaxed();
  const size_t capacity = capacity_.LoadRelaxed();
  size_t need_evict_charge;
  // NOTE: if total_charge > old_usage, there isn't yet enough to evict
  // `total_charge` amount. Even if we only try to evict `old_usage` amount,
  // there's likely something referenced and we would eat CPU looking for
  // enough to evict.
  if (old_usage + total_charge <= capacity || total_charge > old_usage) {
    // Good enough for me (might run over with a race)
    need_evict_charge = 0;
  } else {
    // Try to evict enough space, and maybe some extra
    need_evict_charge = total_charge;
    if (old_usage > capacity) {
      // Not too much to avoid thundering herd while avoiding strict
      // synchronization, such as the compare_exchange used with strict
      // capacity limit.
      need_evict_charge += std::min(capacity / 1024, total_charge) + 1;
    }
    // FIFOCache: and for the slots the table grew by for this insert, which
    // are in old_usage, as FIFOCache did before this table; but no more than
    // the insert overruns capacity by, so an insert evicts only what it
    // needs. Over capacity from racing inserts, that is the overrun plus the
    // charge, still capped as above.
    need_evict_charge += state.slot_charge;
    need_evict_charge =
        std::min(need_evict_charge, old_usage + total_charge - capacity);
  }
  // FIFOCache: an eviction for occupancy alone asks Evict for an entry and
  // no charge, as in the strict case.
  EvictionData data;
  if (need_evict_charge > 0 || UNLIKELY(need_evict_for_occupancy)) {
    // FIFOCache: as in the strict case.
    assert(!state.probation_precharged);
    data.probation_precharge = state.probation_charge;
    static_cast<Table*>(this)->Evict(need_evict_charge,
                                     need_evict_for_occupancy, state, &data);
    state.probation_precharged = data.probation_precharge > 0;
    if (UNLIKELY(data.effort_cap_reached)) {
      eviction_effort_exceeded_count_.FetchAddRelaxed(1);
    }
    // Deal with potential occupancy deficit
    if (UNLIKELY(need_evict_for_occupancy) && data.freed_count == 0) {
      assert(data.freed_charge == 0);
      // Can't meet occupancy requirement
      return false;
    } else {
      // Update occupancy for evictions
      occupancy_.FetchSub(data.freed_count);
    }
  }
  // Track new usage even if we weren't able to evict enough
  usage_.FetchAddRelaxed(total_charge - data.freed_charge);
  // No underflow
  assert(usage_.LoadRelaxed() < SIZE_MAX / 2);
  // Success
  return true;
}

void BaseClockTable::TrackAndReleaseEvictedEntry(ClockHandle* h) {
  bool took_value_ownership = false;
  if (eviction_callback_) {
    // For key reconstructed from hash
    UniqueId64x2 unhashed;
    took_value_ownership = eviction_callback_(h->GetKey(&unhashed, hash_seed_),
                                              static_cast<Cache::Handle*>(h),
                                              h->meta.LoadRelaxed().GetHit());
  }
  if (!took_value_ownership) {
    h->FreeData(allocator_);
  } else {
    // FIFOCache: the key is still the cache's.
    h->FreeHeapKey();
  }
  MarkEmpty(*h);
}

bool BaseClockTable::IsEvictionEffortExceeded(
    const BaseClockTable::EvictionData& data) const {
  auto eviction_effort_cap =
      eec_and_scl_.LoadRelaxed().GetEffectiveEvictionEffortCap();
  // Basically checks whether the ratio of useful effort to wasted effort is
  // too low, with a start-up allowance for wasted effort before any useful
  // effort.
  return (data.freed_count + 1U) * uint64_t{eviction_effort_cap} <=
         data.seen_pinned_count;
}

template <class Table>
Status BaseClockTable::Insert(const ClockHandleBasicData& proto,
                              typename Table::HandleImpl** handle,
                              Cache::Priority priority) {
  using HandleImpl = typename Table::HandleImpl;
  Table& derived = static_cast<Table&>(*this);

  // FIFOCache: refused, without a handle as if inserted and evicted at
  // once.
  if (UNLIKELY(!UsageCanTake(proto.GetTotalCharge()))) {
    if (handle == nullptr) {
      proto.FreeData(allocator_);
      return Status::OK();
    }
    return Status::MemoryLimit("Cache entry too large to account for: " +
                               std::to_string(proto.GetTotalCharge()));
  }

  typename Table::InsertState state;
  derived.StartInsert(state);
  EvictForChargedRing<Table> evict_for_ring(derived, state);
  // FIFOCache: for an insert accepted without a slot (dropped, or standalone
  // with a handle): reverts occupancy and the probation charge its eviction
  // added, puts the ghost tag back, and erases any entry with the key, so
  // the old value doesn't stay visible. Returns whether there was one.
  auto accept_without_slot = [&]() {
    occupancy_.FetchSubRelaxed(1);
    derived.UndoProbationPrecharge(state);
    derived.GhostPutBack(proto, state);
    return derived.EraseReplaced(proto, nullptr);
  };
  // FIFOCache: an entry whose key is in the ghost goes to the resident
  // queue, as does a cache reservation, and any other into the probation
  // queue, where its charge counts toward probation's share. The tag is
  // taken before eviction, which could replace it, and put back on each
  // path below that doesn't put the entry in the table.
  state.reservation = Table::IsReservation(proto, handle != nullptr);
  if (!state.reservation) {
    derived.GhostTake(proto, state);
    if (state.ghost_hit) {
      TEST_SYNC_POINT("FIFOCache::Insert:TookGhostTag");
    }
  }
  state.probation_charge =
      state.ghost_hit || state.reservation ? 0 : proto.GetTotalCharge();

  // Do we have the available occupancy? Optimistically assume we do
  // and deal with it if we don't.
  size_t old_occupancy = occupancy_.FetchAdd(1);
  // Whether we over-committed and need an eviction to make up for it
  bool need_evict_for_occupancy =
      !derived.GrowIfNeeded(old_occupancy + 1, proto.GetTotalCharge(), state);

  // Usage/capacity handling is somewhat different depending on
  // strict_capacity_limit, but mostly pessimistic.
  bool use_standalone_insert = false;
  const size_t total_charge = proto.GetTotalCharge();
  // NOTE: we can use eec_and_scl as eviction_effort_cap below because
  // strict_capacity_limit=true is supposed to disable the limit on eviction
  // effort, and a large value effectively does that.
  if (eec_and_scl_.LoadRelaxed().Get<StrictCapacityLimit>()) {
    Status s = ChargeUsageMaybeEvictStrict<Table>(
        total_charge, need_evict_for_occupancy, state);
    if (!s.ok()) {
      // Revert occupancy
      occupancy_.FetchSubRelaxed(1);
      derived.UndoProbationPrecharge(state);
      derived.GhostPutBack(proto, state);
      // FIFOCache: without a handle, dropped as if inserted and evicted at
      // once, as FIFOCache and LRUCache do, where HyperClockCache returns
      // MemoryLimit. Either way the cache is as it was, an entry with the
      // key included, as in both of them.
      if (handle == nullptr) {
        proto.FreeData(allocator_);
        return Status::OK();
      }
      return s;
    }
  } else {
    // Case strict_capacity_limit == false
    bool success = ChargeUsageMaybeEvictNonStrict<Table>(
        total_charge, need_evict_for_occupancy, state);
    if (!success) {
      const bool replaced = accept_without_slot();
      if (handle == nullptr) {
        // Don't insert the entry but still return ok, as if the entry
        // inserted into cache and evicted immediately.
        proto.FreeData(allocator_);
        return replaced ? Status::OkOverwritten() : Status::OK();
      } else {
        // Need to track usage of fallback standalone insert
        usage_.FetchAddRelaxed(total_charge);
        use_standalone_insert = true;
      }
    }
  }

  if (!use_standalone_insert) {
    // FIFOCache: attempt a table insert, which replaces any entry with the
    // key (see DoInsert), where HyperClockCache aborts if it finds one.

    uint32_t initial_countdown = GetInitialCountdown(priority);
    assert(initial_countdown > 0);

    HandleImpl* e =
        derived.DoInsert(proto, initial_countdown, handle != nullptr, state);

    if (e) {
      // Successfully inserted
      if (handle) {
        *handle = e;
      }
      // FIFOCache: OkOverwritten if it replaced an entry.
      return state.replaced ? Status::OkOverwritten() : Status::OK();
    }
    // Not inserted
    const bool replaced = accept_without_slot();
    // Maybe fall back on standalone insert
    if (handle == nullptr) {
      // Revert usage
      usage_.FetchSubRelaxed(total_charge);
      // No underflow
      assert(usage_.LoadRelaxed() < SIZE_MAX / 2);
      // As if unrefed entry immdiately evicted
      proto.FreeData(allocator_);
      return replaced ? Status::OkOverwritten() : Status::OK();
    }

    use_standalone_insert = true;
  }

  // Run standalone insert
  assert(use_standalone_insert);

  *handle = StandaloneInsert<HandleImpl>(proto);

  // The OkOverwritten status is used to count "redundant" insertions into
  // block cache. This implementation doesn't strictly check for redundant
  // insertions, but we instead are probably interested in how many insertions
  // didn't go into the table (instead "standalone"), which could be redundant
  // Insert or some other reason (use_standalone_insert reasons above).
  // FIFOCache: inserts that replace an entry return it too.
  return Status::OkOverwritten();
}

bool BaseClockTable::ChargeMetadata(size_t bytes, size_t keep) {
  if (metadata_charge_policy_ !=
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    return true;
  }
  if (!eec_and_scl_.LoadRelaxed().Get<StrictCapacityLimit>()) {
    usage_.FetchAddRelaxed(bytes);
    return true;
  }
  const size_t capacity = capacity_.LoadRelaxed();
  size_t old_usage = usage_.LoadRelaxed();
  do {
    if (old_usage > capacity || capacity - old_usage < bytes ||
        capacity - old_usage - bytes < keep) {
      return false;
    }
  } while (!usage_.CasWeakRelaxed(old_usage, old_usage + bytes));
  return true;
}

void BaseClockTable::UnchargeMetadata(size_t bytes) {
  if (metadata_charge_policy_ ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    usage_.FetchSubRelaxed(bytes);
  }
}

void BaseClockTable::Ref(ClockHandle& h) {
  // Increment acquire counter
  SlotMeta old_meta;
  h.meta.Apply(AcquireCounter::PlusTransformPromiseNoOverflow(1), &old_meta);

  assert(old_meta.IsShareable());
  // Must have already had a reference
  assert(old_meta.GetRefcount() > 0);
  (void)old_meta;
}

#ifndef NDEBUG
void BaseClockTable::TEST_RefN(ClockHandle& h, uint32_t n) {
  // Increment acquire counter
  SlotMeta old_meta;
  h.meta.Apply(AcquireCounter::PlusTransformPromiseNoOverflow(n), &old_meta);

  assert(old_meta.IsShareable());
  (void)old_meta;
}

void BaseClockTable::TEST_ReleaseNMinus1(ClockHandle* h, uint32_t n) {
  assert(n > 0);

  // Like n-1 Releases, but assumes one more will happen in the caller to take
  // care of anything like erasing an unreferenced, invisible entry.
  SlotMeta old_meta;
  h->meta.Apply(ReleaseCounter::PlusTransformPromiseNoOverflow(n - 1),
                &old_meta);
  assert(old_meta.IsShareable());
  (void)old_meta;
}
#endif

template <class Table>
ClockCacheShard<Table>::ClockCacheShard(
    size_t capacity, bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy,
    MemoryAllocator* allocator,
    const Cache::EvictionCallback* eviction_callback, const uint32_t* hash_seed,
    const typename Table::Opts& opts)
    : CacheShardBase(metadata_charge_policy),
      table_(capacity, strict_capacity_limit, metadata_charge_policy, allocator,
             eviction_callback, hash_seed, opts) {
  // FIFOCache: a shard may be smaller than its starting table's charge
  // (upstream asserts otherwise); it then holds little until capacity grows.
}

template <class Table>
void ClockCacheShard<Table>::EraseUnRefEntries() {
  table_.EraseUnRefEntries();
}

template <class Table>
void ClockCacheShard<Table>::ApplyToSomeEntries(
    const std::function<void(const Slice& key, Cache::ObjectPtr value,
                             size_t charge,
                             const Cache::CacheItemHelper* helper)>& callback,
    size_t average_entries_per_lock, size_t* state) {
  // The state will be a simple index into the table. Even with a dynamic
  // hyper clock cache, entries will generally stay in their existing
  // slots, so we don't need to be aware of the high-level organization
  // that makes lookup efficient.
  // FIFOCache: up to the claimed slots (see GetClaimedLength).
  size_t length = table_.GetClaimedLength();

  assert(average_entries_per_lock > 0);

  size_t index_begin = *state;
  size_t index_end = index_begin + average_entries_per_lock;
  if (index_end >= length) {
    // Going to end.
    index_end = length;
    *state = SIZE_MAX;
  } else {
    *state = index_end;
  }

  auto hash_seed = table_.GetHashSeed();
  ConstApplyToEntriesRange(
      [callback, hash_seed](const HandleImpl& h) {
        UniqueId64x2 unhashed;
        callback(h.GetKey(&unhashed, hash_seed), h.value, h.GetCharge(),
                 h.GetHelper());
      },
      table_.HandlePtr(index_begin), table_.HandlePtr(index_end), false);
}

template <class Table>
void ClockCacheShard<Table>::SetCapacity(size_t capacity) {
  table_.SetCapacity(capacity);
  // FIFOCache: and evicts now, rather than at the next Insert.
  table_.EvictOverCapacity();
}

template <class Table>
void ClockCacheShard<Table>::SetStrictCapacityLimit(
    bool strict_capacity_limit) {
  table_.SetStrictCapacityLimit(strict_capacity_limit);
  // next Insert will take care of any necessary evictions
}

template <class Table>
Status ClockCacheShard<Table>::Insert(const Slice& key,
                                      const UniqueId64x2& hashed_key,
                                      Cache::ObjectPtr value,
                                      const Cache::CacheItemHelper* helper,
                                      size_t charge, HandleImpl** handle,
                                      Cache::Priority priority) {
  ClockHandleBasicData proto;
  proto.hashed_key = hashed_key;
  proto.value = value;
  proto.total_charge = charge;
  if (!proto.SetKeyAndHelper(key, helper, table_.GetMetadataChargePolicy())) {
    return Status::MemoryLimit("FIFOCache could not copy the key");
  }
  Status s = table_.template Insert<Table>(proto, handle, priority);
  if (!s.ok()) {
    // FIFOCache: the table took neither the value nor the key.
    proto.FreeHeapKey();
  }
  return s;
}

template <class Table>
typename Table::HandleImpl* ClockCacheShard<Table>::CreateStandalone(
    const Slice& key, const UniqueId64x2& hashed_key, Cache::ObjectPtr obj,
    const Cache::CacheItemHelper* helper, size_t charge, bool allow_uncharged) {
  ClockHandleBasicData proto;
  proto.hashed_key = hashed_key;
  proto.value = obj;
  proto.total_charge = charge;
  if (!proto.SetKeyAndHelper(key, helper, table_.GetMetadataChargePolicy())) {
    return nullptr;
  }
  HandleImpl* h =
      table_.template CreateStandalone<Table>(proto, allow_uncharged);
  if (h == nullptr) {
    // FIFOCache: the caller keeps the value, and the key is freed.
    proto.FreeHeapKey();
  }
  return h;
}

template <class Table>
typename ClockCacheShard<Table>::HandleImpl* ClockCacheShard<Table>::Lookup(
    const Slice& key, const UniqueId64x2& hashed_key) {
  return table_.Lookup(hashed_key,
                       key.size() == kCacheKeySize ? nullptr : &key);
}

template <class Table>
bool ClockCacheShard<Table>::Ref(HandleImpl* h) {
  if (h == nullptr) {
    return false;
  }
  table_.Ref(*h);
  return true;
}

template <class Table>
bool ClockCacheShard<Table>::Release(HandleImpl* handle, bool useful,
                                     bool erase_if_last_ref) {
  if (handle == nullptr) {
    return false;
  }
  return table_.Release(handle, useful, erase_if_last_ref);
}

#ifndef NDEBUG
template <class Table>
void ClockCacheShard<Table>::TEST_RefN(HandleImpl* h, uint32_t n) {
  table_.TEST_RefN(*h, n);
}

template <class Table>
void ClockCacheShard<Table>::TEST_ReleaseN(HandleImpl* h, uint32_t n) {
  table_.TEST_ReleaseN(h, n);
}
#endif

template <class Table>
bool ClockCacheShard<Table>::Release(HandleImpl* handle,
                                     bool erase_if_last_ref) {
  return Release(handle, /*useful=*/true, erase_if_last_ref);
}

template <class Table>
void ClockCacheShard<Table>::Erase(const Slice& key,
                                   const UniqueId64x2& hashed_key) {
  table_.Erase(hashed_key, key.size() == kCacheKeySize ? nullptr : &key);
}

template <class Table>
size_t ClockCacheShard<Table>::GetUsage() const {
  return table_.GetUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetStandaloneUsage() const {
  return table_.GetStandaloneUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetCapacity() const {
  return table_.GetCapacity();
}

template <class Table>
size_t ClockCacheShard<Table>::GetPinnedUsage() const {
  // Computes the pinned usage by scanning the whole hash table. This
  // is slow, but avoids keeping an exact counter on the clock usage,
  // i.e., the number of not externally referenced elements.
  // Why avoid this counter? Because Lookup removes elements from the clock
  // list, so it would need to update the pinned usage every time,
  // which creates additional synchronization costs.
  // FIFOCache: without the pinned entries' slots, which are part of the
  // table's charge, as FIFOCache did before this table.
  size_t table_pinned_usage = 0;
  ConstApplyToEntriesRange(
      [&table_pinned_usage](const HandleImpl& h) {
        TEST_SYNC_POINT_CALLBACK("FIFOCache::GetPinnedUsage:Held",
                                 const_cast<HandleImpl*>(&h));
        SlotMeta meta = h.meta.LoadRelaxed();
        uint32_t refcount = meta.GetRefcount();
        // Holding one ref for ConstApplyToEntriesRange
        assert(refcount > 0);
        if (refcount > 1) {
          table_pinned_usage += h.GetTotalCharge();
        }
      },
      table_.HandlePtr(0), table_.HandlePtr(table_.GetClaimedLength()), true);

  return table_pinned_usage + table_.GetStandaloneUsage();
}

template <class Table>
size_t ClockCacheShard<Table>::GetOccupancyCount() const {
  return table_.GetOccupancy();
}

template <class Table>
size_t ClockCacheShard<Table>::GetOccupancyLimit() const {
  return table_.GetOccupancyLimit();
}

template <class Table>
size_t ClockCacheShard<Table>::GetTableAddressCount() const {
  return table_.GetTableSize();
}

// Explicit instantiation
template class ClockCacheShard<AutoHyperClockTable>;

template <class Table>
BaseHyperClockCache<Table>::BaseHyperClockCache(
    const HyperClockCacheOptions& opts, size_t max_capacity)
    : ShardedCache<ClockCacheShard<Table>>(opts) {
  // TODO: should not need to go through two levels of pointer indirection to
  // get to table entries
  size_t per_shard = this->GetPerShardCapacity();
  // FIFOCache: max_capacity split across shards as capacity is, rounding
  // up.
  const size_t num_shards = this->GetNumShards();
  const size_t per_shard_max =
      max_capacity / num_shards + (max_capacity % num_shards != 0 ? 1 : 0);
  MemoryAllocator* alloc = this->memory_allocator();
  this->InitShards([&](Shard* cs) {
    typename Table::Opts table_opts{opts};
    table_opts.reserved_capacity = per_shard_max;
    new (cs) Shard(per_shard, opts.strict_capacity_limit,
                   opts.metadata_charge_policy, alloc,
                   &this->eviction_callback_, &this->hash_seed_, table_opts);
  });
}

template <class Table>
Cache::ObjectPtr BaseHyperClockCache<Table>::Value(Handle* handle) {
  return static_cast<const typename Table::HandleImpl*>(handle)->value;
}

template <class Table>
size_t BaseHyperClockCache<Table>::GetCharge(Handle* handle) const {
  return static_cast<const typename Table::HandleImpl*>(handle)->GetCharge();
}

template <class Table>
const Cache::CacheItemHelper* BaseHyperClockCache<Table>::GetCacheItemHelper(
    Handle* handle) const {
  auto h = static_cast<const typename Table::HandleImpl*>(handle);
  return h->GetHelper();
}

template <class Table>
void BaseHyperClockCache<Table>::ApplyToHandle(
    Cache* cache, Handle* handle,
    const std::function<void(const Slice& key, Cache::ObjectPtr value,
                             size_t charge, const CacheItemHelper* helper)>&
        callback) {
  BaseHyperClockCache<Table>* cache_ptr =
      static_cast<BaseHyperClockCache<Table>*>(cache);
  auto h = static_cast<const typename Table::HandleImpl*>(handle);
  UniqueId64x2 unhashed;
  auto hash_seed = cache_ptr->GetShard(h->GetHash()).GetTable().GetHashSeed();
  callback(h->GetKey(&unhashed, hash_seed), h->value, h->GetCharge(),
           h->GetHelper());
}

namespace {

bool IsSlotOccupied(const ClockHandle& h) {
  return !h.meta.LoadRelaxed().IsEmpty();
}
}  // namespace

// NOTE: GCC might warn about subobject linkage if this is in anon namespace
template <size_t N = 500>
class LoadVarianceStats {
 public:
  std::string Report() const {
    return "Overall " + PercentStr(positive_count_, samples_) + " (" +
           std::to_string(positive_count_) + "/" + std::to_string(samples_) +
           "), Min/Max/Window = " + PercentStr(min_, N) + "/" +
           PercentStr(max_, N) + "/" + std::to_string(N) +
           ", MaxRun{Pos/Neg} = " + std::to_string(max_pos_run_) + "/" +
           std::to_string(max_neg_run_);
  }

  void Add(bool positive) {
    recent_[samples_ % N] = positive;
    if (positive) {
      ++positive_count_;
      ++cur_pos_run_;
      max_pos_run_ = std::max(max_pos_run_, cur_pos_run_);
      cur_neg_run_ = 0;
    } else {
      ++cur_neg_run_;
      max_neg_run_ = std::max(max_neg_run_, cur_neg_run_);
      cur_pos_run_ = 0;
    }
    ++samples_;
    if (samples_ >= N) {
      size_t count_set = recent_.count();
      max_ = std::max(max_, count_set);
      min_ = std::min(min_, count_set);
    }
  }

 private:
  size_t max_ = 0;
  size_t min_ = N;
  size_t positive_count_ = 0;
  size_t samples_ = 0;
  size_t max_pos_run_ = 0;
  size_t cur_pos_run_ = 0;
  size_t max_neg_run_ = 0;
  size_t cur_neg_run_ = 0;
  std::bitset<N> recent_;

  static std::string PercentStr(size_t a, size_t b) {
    if (b == 0) {
      return "??%";
    } else {
      return std::to_string(uint64_t{100} * a / b) + "%";
    }
  }
};

template <class Table>
void BaseHyperClockCache<Table>::ReportProblems(
    const std::shared_ptr<Logger>& info_log) const {
  if (info_log->GetInfoLogLevel() <= InfoLogLevel::DEBUG_LEVEL) {
    LoadVarianceStats slot_stats;
    uint64_t eviction_effort_exceeded_count = 0;
    this->ForEachShard([&](const BaseHyperClockCache<Table>::Shard* shard) {
      size_t count = shard->GetTableAddressCount();
      for (size_t i = 0; i < count; ++i) {
        slot_stats.Add(IsSlotOccupied(*shard->GetTable().HandlePtr(i)));
      }
      eviction_effort_exceeded_count +=
          shard->GetTable().GetEvictionEffortExceededCount();
    });
    ROCKS_LOG_AT_LEVEL(info_log, InfoLogLevel::DEBUG_LEVEL,
                       "Slot occupancy stats: %s", slot_stats.Report().c_str());
    ROCKS_LOG_AT_LEVEL(info_log, InfoLogLevel::DEBUG_LEVEL,
                       "Eviction effort exceeded: %" PRIu64,
                       eviction_effort_exceeded_count);
  }
}

// =======================================================================
//                             AutoHyperClockCache
// =======================================================================

// See AutoHyperClockTable::length_info_ etc. for how the linear hashing
// metadata is encoded. Here are some example values:
//
// Used length  | min shift  | threshold  | max shift
// 2            | 1          | 0          | 1
// 3            | 1          | 1          | 2
// 4            | 2          | 0          | 2
// 5            | 2          | 1          | 3
// 6            | 2          | 2          | 3
// 7            | 2          | 3          | 3
// 8            | 3          | 0          | 3
// 9            | 3          | 1          | 4
// ...
// Note:
// * min shift = floor(log2(used length))
// * max shift = ceil(log2(used length))
// * used length == (1 << shift) + threshold
// Also, shift=0 is never used in practice, so is reserved for "unset"

namespace {

inline int LengthInfoToMinShift(uint64_t length_info) {
  int mask_shift = BitwiseAnd(length_info, int{255});
  assert(mask_shift <= 63);
  assert(mask_shift > 0);
  return mask_shift;
}

inline size_t LengthInfoToThreshold(uint64_t length_info) {
  return static_cast<size_t>(length_info >> 8);
}

inline size_t LengthInfoToUsedLength(uint64_t length_info) {
  size_t threshold = LengthInfoToThreshold(length_info);
  int shift = LengthInfoToMinShift(length_info);
  assert(threshold < (size_t{1} << shift));
  size_t used_length = (size_t{1} << shift) + threshold;
  assert(used_length >= 2);
  return used_length;
}

inline uint64_t UsedLengthToLengthInfo(size_t used_length) {
  assert(used_length >= 2);
  int shift = FloorLog2(used_length);
  uint64_t threshold = BottomNBits(used_length, shift);
  uint64_t length_info =
      (uint64_t{threshold} << 8) + static_cast<uint64_t>(shift);
  assert(LengthInfoToUsedLength(length_info) == used_length);
  assert(LengthInfoToMinShift(length_info) == shift);
  assert(LengthInfoToThreshold(length_info) == threshold);
  return length_info;
}

// Avoid potential initialization order race with port::kPageSize
constexpr size_t kPresumedPageSize = 4096;

inline size_t GetStartingLength(size_t capacity) {
  if (capacity > kPresumedPageSize) {
    // Start with one memory page
    return kPresumedPageSize / sizeof(AutoHyperClockTable::HandleImpl);
  } else {
    // Mostly to make unit tests happy
    return 4;
  }
}

inline size_t GetHomeIndex(uint64_t hash, int shift) {
  return static_cast<size_t>(BottomNBits(hash, shift));
}

inline void GetHomeIndexAndShift(uint64_t length_info, uint64_t hash,
                                 size_t* home, int* shift) {
  int min_shift = LengthInfoToMinShift(length_info);
  size_t threshold = LengthInfoToThreshold(length_info);
  bool extra_shift = GetHomeIndex(hash, min_shift) < threshold;
  *home = GetHomeIndex(hash, min_shift + extra_shift);
  *shift = min_shift + extra_shift;
  assert(*home < LengthInfoToUsedLength(length_info));
}

// Helper function for Lookup
// FIFOCache: key as for ClockHandleBasicData::MatchesKey, read only with
// the reference held.
inline bool MatchAndRef(const UniqueId64x2* hashed_key, const Slice* key,
                        const ClockHandle& h, int shift = 0, size_t home = 0,
                        bool* full_match_or_unknown = nullptr) {
  // Must be at least something to match
  assert(hashed_key || shift > 0);

  SlotMeta old_meta, new_meta;
  // (Optimistically) increment acquire counter.
  auto add_acquire = AcquireCounter::PlusTransformPromiseNoOverflow(1);
  h.meta.Apply(add_acquire, &old_meta, &new_meta);
  // Check if it's a referencable (sharable) entry
  if (!old_meta.IsShareable()) {
    // For non-sharable states, incrementing the acquire counter has no effect
    // so we don't need to undo it. Furthermore, we cannot safely undo
    // it because we did not acquire a read reference to lock the
    // entry in a Shareable state.
    if (full_match_or_unknown) {
      *full_match_or_unknown = true;
    }
    return false;
  }
  // Else acquired a read reference
  assert(new_meta.GetRefcount() > 0);
  if (hashed_key && h.MatchesKey(*hashed_key, key) &&
      LIKELY(old_meta.IsVisible())) {
    // Match on full key, visible
    if (full_match_or_unknown) {
      *full_match_or_unknown = true;
    }
    return true;
  } else if (shift > 0 && home == BottomNBits(h.hashed_key[1], shift)) {
    // NOTE: upper 32 bits of hashed_key[0] is used for sharding
    // Match on home address, possibly invisible
    if (full_match_or_unknown) {
      *full_match_or_unknown = false;
    }
    return true;
  } else {
    // Mismatch. Pretend we never took the reference
    Unref(h);
    if (full_match_or_unknown) {
      *full_match_or_unknown = false;
    }
    return false;
  }
}

using NextWithShift = AutoHyperClockTable::HandleImpl::NextWithShift;

// Assumes a chain rewrite lock prevents concurrent modification of
// these chain pointers
void UpgradeShiftsOnRange(AutoHyperClockTable::HandleImpl* arr,
                          size_t& frontier,
                          NextWithShift stop_before_or_new_tail, int old_shift,
                          int new_shift) {
  assert(frontier != SIZE_MAX);
  assert(new_shift == old_shift + 1);
  (void)old_shift;
  (void)new_shift;
  for (;;) {
    NextWithShift next_with_shift = arr[frontier].chain_next_with_shift.Load();
    assert(next_with_shift.GetShift() == old_shift);
    if (next_with_shift == stop_before_or_new_tail) {
      // Stopping at entry with pointer matching "stop before"
      assert(!next_with_shift.IsEnd());
      return;
    }
    if (next_with_shift.IsEnd()) {
      // Also update tail to new tail
      assert(stop_before_or_new_tail.IsEnd());
      arr[frontier].chain_next_with_shift.Store(stop_before_or_new_tail);
      // Mark nothing left to upgrade
      frontier = SIZE_MAX;
      return;
    }
    // Next is another entry to process, so upgrade and advance frontier
    arr[frontier].chain_next_with_shift.Apply(
        NextWithShift::Shift::PlusTransformPromiseNoOverflow(1U));
    assert(next_with_shift.GetShift() + 1 == new_shift);
    frontier = next_with_shift.GetNext();
  }
}

size_t CalcOccupancyLimit(size_t used_length) {
  return static_cast<size_t>(used_length * AutoHyperClockTable::kMaxLoadFactor +
                             0.999);
}

// FIFOCache: see AutoHyperClockTable::ghost_stamp_shift_. Leaves the stamp's
// top bit spare, so a tag that stopped counting doesn't read as counting
// again until its age is within kGhostStampSlack steps of wrapping, unless
// the shard's entry count grows past its age first.
int GhostStampShift(size_t occupancy_limit) {
  int shift = 0;
  while ((occupancy_limit >> shift) >= (size_t{1} << 15)) {
    ++shift;
  }
  return shift;
}

}  // namespace

// An RAII wrapper for locking a chain of entries (flag bit on the head)
// so that there is only one thread allowed to remove entries from the
// chain, or to rewrite it by splitting for Grow. Without the lock,
// all lookups and insertions at the head can proceed wait-free.
// The class also provides functions for safely manipulating the head pointer
// while holding the lock--or wanting to should it become non-empty.
//
// The flag bits on the head are such that the head cannot be locked if it
// is an empty chain, so that a "blind" FetchOr will try to lock a non-empty
// chain but have no effect on an empty chain. When a potential rewrite
// operation see an empty head pointer, there is no need to lock as the
// operation is a no-op. However, there are some cases such as CAS-update
// where locking might be required after initially not being needed, if the
// operation is forced to revisit the head pointer.
class AutoHyperClockTable::ChainRewriteLock {
 public:
  using HandleImpl = AutoHyperClockTable::HandleImpl;

  // Acquire lock if head of h is not an end
  explicit ChainRewriteLock(HandleImpl* h, RelaxedAtomic<uint64_t>& yield_count)
      : head_ptr_(&h->head_next_with_shift) {
    Acquire(yield_count);
  }

  // RAII wrap existing lock held (or end)
  explicit ChainRewriteLock(HandleImpl* h,
                            RelaxedAtomic<uint64_t>& /*yield_count*/,
                            NextWithShift already_locked_or_end)
      : head_ptr_(&h->head_next_with_shift) {
    saved_head_ = already_locked_or_end;
    // already locked or end
    assert(saved_head_.IsLocked());
  }

  ~ChainRewriteLock() {
    if (!IsEnd()) {
      // Release lock
      NextWithShift old;
      head_ptr_->Apply(NextWithShift::LockedFlag::ClearTransform(), &old);
      assert(old.IsLockedNotEnd());
    }
  }

  void Reset(HandleImpl* h, RelaxedAtomic<uint64_t>& yield_count) {
    this->~ChainRewriteLock();
    new (this) ChainRewriteLock(h, yield_count);
  }

  // Expected current state, assuming no parallel updates.
  NextWithShift GetSavedHead() const { return saved_head_; }

  bool CasUpdate(NextWithShift next_with_shift,
                 RelaxedAtomic<uint64_t>& yield_count) {
    NextWithShift new_head =
        next_with_shift.With<NextWithShift::LockedFlag>(true);
    NextWithShift expected = GetSavedHead();
    bool success = head_ptr_->CasStrong(expected, new_head);
    if (success) {
      // Ensure IsEnd() is kept up-to-date, including for dtor
      saved_head_ = new_head;
    } else {
      // Parallel update to head, such as Insert()
      if (IsEnd()) {
        // Didn't previously hold a lock
        if (expected.IsEnd()) {
          // Still don't need to
          saved_head_ = expected;
        } else {
          // Need to acquire lock before proceeding
          Acquire(yield_count);
        }
      } else {
        // Parallel update must preserve our lock
        assert(expected.IsLockedNotEnd());
        saved_head_ = expected;
      }
    }
    return success;
  }

  bool IsEnd() const { return saved_head_.IsEnd(); }

 private:
  void Acquire(RelaxedAtomic<uint64_t>& yield_count) {
    for (;;) {
      // Acquire removal lock on the chain
      NextWithShift old_head;
      head_ptr_->Apply(NextWithShift::LockedFlag::SetTransform(), &old_head,
                       &saved_head_);
      if (!old_head.IsLockedNotEnd()) {
        // Either acquired the lock or lock not needed (end)
        assert(old_head.IsEnd() == old_head.IsLocked());
        break;
      }
      // NOTE: one of the few yield-wait loops, which is rare enough in practice
      // for its performance to be insignificant. (E.g. using C++20 atomic
      // wait/notify would likely be worse because of wasted notify costs.)
      yield_count.FetchAddRelaxed(1);
      std::this_thread::yield();
    }
  }

  BitFieldsAtomic<NextWithShift>* head_ptr_;
  NextWithShift saved_head_;
};

AutoHyperClockTable::AutoHyperClockTable(
    size_t capacity, bool strict_capacity_limit,
    CacheMetadataChargePolicy metadata_charge_policy,
    MemoryAllocator* allocator,
    const Cache::EvictionCallback* eviction_callback, const uint32_t* hash_seed,
    const Opts& opts)
    : BaseClockTable(capacity, strict_capacity_limit, opts.eviction_effort_cap,
                     metadata_charge_policy, allocator, eviction_callback,
                     hash_seed),
      array_(sizeof(HandleImpl) *
             CalcMaxUsableLength(std::max(capacity, opts.reserved_capacity))),
      length_info_(UsedLengthToLengthInfo(GetStartingLength(capacity))),
      occupancy_limit_(
          CalcOccupancyLimit(LengthInfoToUsedLength(length_info_.Load()))),
      grow_frontier_(GetTableSize()),
      clock_pointer_mask_(
          BottomNBits(UINT64_MAX, LengthInfoToMinShift(length_info_.Load()))),
      // FIFOCache: the probation words and ghost tags, one each per slot,
      // and the queue.
      probation_words_(sizeof(RelaxedAtomic<uint32_t>) * array_.Count()),
      ghost_(sizeof(RelaxedAtomic<uint32_t>) * array_.Count()),
      ghost_stamp_shift_(
          GhostStampShift(CalcOccupancyLimit(CalcMaxUsableLength(capacity)))),
      probation_(ProbationRingCapacity(occupancy_limit_.LoadRelaxed()),
                 ProbationRingCapacity(CalcOccupancyLimit(array_.Count()))) {
  // FIFOCache: and the probation words' and ghost's mappings.
  if (array_.Get() == nullptr || probation_words_.Get() == nullptr ||
      ghost_.Get() == nullptr) {
    fprintf(stderr, "Anonymous mmap for RocksDB FIFOCache failed. Aborting.\n");
    std::terminate();
  }
  if (metadata_charge_policy ==
      CacheMetadataChargePolicy::kFullChargeCacheMetadata) {
    // NOTE: ignoring page boundaries for simplicity
    // FIFOCache: with the probation words, ghost tags and the first
    // probation ring.
    usage_.FetchAddRelaxed(GetMetadataCharge());
  }

  static_assert(sizeof(HandleImpl) == 64U,
                "Expecting size / alignment with common cache line size");

  // Populate head pointers
  uint64_t length_info = length_info_.Load();
  int min_shift = LengthInfoToMinShift(length_info);
  int max_shift = min_shift + 1;
  size_t major = uint64_t{1} << min_shift;
  size_t used_length = GetTableSize();

  assert(major <= used_length);
  assert(used_length <= major * 2);

  // Initialize the initial usable set of slots. This slightly odd iteration
  // order makes it easier to get the correct shift amount on each head.
  for (size_t i = 0; i < major; ++i) {
#ifndef NDEBUG
    int shift;
    size_t home;
#endif
    if (major + i < used_length) {
      array_[i].head_next_with_shift.StoreRelaxed(
          NextWithShift::MakeEnd(i, max_shift));
      array_[major + i].head_next_with_shift.StoreRelaxed(
          NextWithShift::MakeEnd(major + i, max_shift));
#ifndef NDEBUG  // Extra invariant checking
      GetHomeIndexAndShift(length_info, i, &home, &shift);
      assert(home == i);
      assert(shift == max_shift);
      GetHomeIndexAndShift(length_info, major + i, &home, &shift);
      assert(home == major + i);
      assert(shift == max_shift);
#endif
    } else {
      array_[i].head_next_with_shift.StoreRelaxed(
          NextWithShift::MakeEnd(i, min_shift));
#ifndef NDEBUG  // Extra invariant checking
      GetHomeIndexAndShift(length_info, i, &home, &shift);
      assert(home == i);
      assert(shift == min_shift);
      GetHomeIndexAndShift(length_info, major + i, &home, &shift);
      assert(home == i);
      assert(shift == min_shift);
#endif
    }
  }
}

AutoHyperClockTable::~AutoHyperClockTable() {
  // As usual, destructor assumes there are no references or active operations
  // on any slot/element in the table.

  // It's possible that there were not enough Insert() after final concurrent
  // Grow to ensure length_info_ (published GetTableSize()) is fully up to
  // date. Probe for first unused slot to ensure we see the whole structure.
  size_t used_end = GetTableSize();
  while (used_end < array_.Count() &&
         array_[used_end].head_next_with_shift.LoadRelaxed() !=
             HandleImpl::kUnusedMarker) {
    used_end++;
  }
#ifndef NDEBUG
  // FIFOCache: probation usage is the charge of the entries with
  // kInProbation set, and probation pinned usage of those that also have
  // kSeenPinned; only entries have them set, and the resident count is the
  // number of the other entries.
  size_t probation_sum = 0;
  size_t probation_pinned_sum = 0;
  int64_t resident_sum = 0;
  for (size_t i = 0; i < used_end; i++) {
    const HandleImpl& h = array_[i];
    const bool shareable = h.meta.LoadRelaxed().IsShareable();
    const uint32_t word = probation_words_[i].LoadRelaxed();
    if (word & kInProbation) {
      assert(shareable);
      probation_sum += h.total_charge;
      if (word & kSeenPinned) {
        probation_pinned_sum += h.total_charge;
      }
    } else {
      assert((word & kSeenPinned) == 0);
      if (shareable && !(word & kReservation)) {
        resident_sum++;
      }
    }
  }
  if (probation_sum != probation_usage_.LoadRelaxed() ||
      probation_pinned_sum != probation_pinned_usage_.LoadRelaxed() ||
      resident_sum != resident_count_.LoadRelaxed()) {
    fprintf(stderr,
            "FIFOCache: probation usage %zu, entries in probation %zu; "
            "pinned %zu, seen pinned %zu; resident count %lld, resident "
            "entries %lld\n",
            probation_usage_.LoadRelaxed(), probation_sum,
            probation_pinned_usage_.LoadRelaxed(), probation_pinned_sum,
            static_cast<long long>(resident_count_.LoadRelaxed()),
            static_cast<long long>(resident_sum));
    assert(false);
  }
#endif  // !NDEBUG
  // This check can be extra expensive for a cache that is just created,
  // maybe used for a small number of entries, as in a unit test, and then
  // destroyed. Only do this in rare modes. REVISED: Don't scan the whole mmap,
  // just a reasonable frontier past what we expect to have written.
#ifdef MUST_FREE_HEAP_ALLOCATIONS
  for (size_t i = used_end; i < array_.Count() && i < used_end + 64U; i++) {
    assert(array_[i].head_next_with_shift.LoadRelaxed() ==
           HandleImpl::kUnusedMarker);
    assert(array_[i].chain_next_with_shift.LoadRelaxed() ==
           HandleImpl::kUnusedMarker);
    assert(array_[i].meta.LoadRelaxed() == SlotMeta{});
  }
#endif          // MUST_FREE_HEAP_ALLOCATIONS
#ifndef NDEBUG  // Extra invariant checking
  std::vector<bool> was_populated(used_end);
  std::vector<bool> was_pointed_to(used_end);
#endif  // !NDEBUG
  for (size_t i = 0; i < used_end; i++) {
    HandleImpl& h = array_[i];
    SlotMeta meta = h.meta.LoadRelaxed();
    if (meta.IsShareable()) {
      // NOTE: Reaching here invisible is rare but possible
      assert(meta.GetRefcount() == 0);
      h.FreeData(allocator_);
#ifndef NDEBUG  // Extra invariant checking
      usage_.FetchSubRelaxed(h.total_charge);
      occupancy_.FetchSubRelaxed(1U);
      was_populated[i] = true;
      if (!h.chain_next_with_shift.LoadRelaxed().IsEnd()) {
        assert(!h.chain_next_with_shift.LoadRelaxed().IsLocked());
        size_t next = h.chain_next_with_shift.LoadRelaxed().GetNext();
        assert(!was_pointed_to[next]);
        was_pointed_to[next] = true;
      }
#endif  // !NDEBUG
    } else {
      // Should be no transient "under construction" states unless a thread
      // was killed or we are being destructed while another thread is still
      // operating on the structure
      assert(meta.IsEmpty());
    }
#ifndef NDEBUG  // Extra invariant checking
    if (!h.head_next_with_shift.LoadRelaxed().IsEnd()) {
      size_t next = h.head_next_with_shift.LoadRelaxed().GetNext();
      assert(!was_pointed_to[next]);
      was_pointed_to[next] = true;
    }
#endif  // !NDEBUG
  }
#ifndef NDEBUG  // Extra invariant checking
  // This check is not perfect, but should detect most reasonable cases
  // of abandonned or floating entries, etc.  (A floating cycle would not
  // be reported as bad.)
  for (size_t i = 0; i < used_end; i++) {
    if (was_populated[i]) {
      assert(was_pointed_to[i]);
    } else {
      assert(!was_pointed_to[i]);
    }
  }
#endif  // !NDEBUG

  // FIFOCache: metadata charging follows the slots Grow claimed, and the
  // probation words and rings.
  assert(usage_.LoadRelaxed() == 0 ||
         usage_.LoadRelaxed() == GetMetadataCharge());
  assert(occupancy_.LoadRelaxed() == 0);
}

size_t AutoHyperClockTable::GetTableSize() const {
  return LengthInfoToUsedLength(length_info_.Load());
}

size_t AutoHyperClockTable::GetClaimedLength() const {
  return std::min(grow_frontier_.LoadRelaxed(), array_.Count());
}

size_t AutoHyperClockTable::GetMetadataCharge() const {
  return GetClaimedLength() *
             (sizeof(HandleImpl) + 2 * sizeof(RelaxedAtomic<uint32_t>)) +
         probation_.LiveCellCount() * sizeof(ProbationQueue::Cell);
}

size_t AutoHyperClockTable::GetOccupancyLimit() const {
  return occupancy_limit_.LoadRelaxed();
}

void AutoHyperClockTable::StartInsert(InsertState& state) {
  state.saved_length_info = length_info_.Load();
}

// Because we have linked lists, bugs or even hardware errors can make it
// possible to create a cycle, which would lead to infinite loop.
// Furthermore, when we have retry cases in the code, we want to be sure
// these are not (and do not become) spin-wait loops. Given the assumption
// of quality hashing and the infeasibility of consistently recurring
// concurrent modifications to an entry or chain, we can safely bound the
// number of loop iterations in feasible operation, whether following chain
// pointers or retrying with some backtracking. A smaller limit is used for
// stress testing, to detect potential issues such as cycles or spin-waits,
// and a larger limit is used to break cycles should they occur in production.
#define CHECK_TOO_MANY_ITERATIONS(i) \
  {                                  \
    assert(i < 768);                 \
    if (UNLIKELY(i >= 4096)) {       \
      std::terminate();              \
    }                                \
  }

bool AutoHyperClockTable::GrowIfNeeded(size_t new_occupancy,
                                       size_t entry_charge,
                                       InsertState& state) {
  // new_occupancy has taken into account other threads that are also trying
  // to insert, so as soon as we see sufficient *published* usable size, we
  // can declare success even if we aren't the one that grows the table.
  // However, there's an awkward state where other threads own growing the
  // table to sufficient usable size, but the udpated size is not yet
  // published. If we wait, then that likely slows the ramp-up cache
  // performance. If we unblock ourselves by ensuring we grow by at least one
  // slot, we could technically overshoot required size by number of parallel
  // threads accessing block cache. On balance considering typical cases and
  // the modest consequences of table being slightly too large, the latter
  // seems preferable.
  //
  // So if the published occupancy limit is too small, we unblock ourselves
  // by committing to growing the table by at least one slot. Also note that
  // we might need to grow more than once to actually increase the occupancy
  // limit (due to max load factor < 1.0)

  while (UNLIKELY(new_occupancy > occupancy_limit_.LoadRelaxed())) {
    // At this point we commit the thread to growing unless we've reached the
    // limit (returns false).
    if (!Grow(state, entry_charge)) {
      return false;
    }
  }
  // Success (didn't need to grow, or did successfully)
  return true;
}

bool AutoHyperClockTable::Grow(InsertState& state, size_t entry_charge) {
  // FIFOCache: the slot, its probation word and ghost tag, charged here,
  // where HyperClockCache charges a slot once length_info_ includes it, so
  // the strict limit can turn the growth down.
  constexpr size_t kSlotMetadata =
      sizeof(HandleImpl) + 2 * sizeof(RelaxedAtomic<uint32_t>);
  // A table at its reservation stays there, so it isn't charged and
  // refunded on every insert.
  if (grow_frontier_.LoadRelaxed() >= array_.Count()) {
    return false;
  }
  if (!ChargeMetadata(kSlotMetadata, entry_charge)) {
    return false;
  }
  // Allocate the next grow slot
  size_t grow_home = grow_frontier_.FetchAddRelaxed(1);
  if (grow_home >= array_.Count()) {
    // Can't grow any more.
    // (Tested by unit test ClockCacheTest/Limits)
    // Make sure we don't overflow grow_frontier_ by reaching here repeatedly
    grow_frontier_.StoreRelaxed(array_.Count());
    UnchargeMetadata(kSlotMetadata);  // FIFOCache
    return false;
  }
  // FIFOCache
  if (metadata_charge_policy_ == kFullChargeCacheMetadata) {
    state.slot_charge += kSlotMetadata;
  }
  TEST_SYNC_POINT_CALLBACK("FIFOCache::Grow:Claimed", &grow_home);
#ifdef COERCE_CONTEXT_SWITCH
  // This is useful in reproducing concurrency issues in Grow()
  while (Random::GetTLSInstance()->OneIn(2)) {
    std::this_thread::yield();
  }
#endif
  // Basically, to implement https://en.wikipedia.org/wiki/Linear_hashing
  // entries that belong in a new chain starting at grow_home will be
  // split off from the chain starting at old_home, which is computed here.
  int old_shift = FloorLog2(grow_home);
  size_t old_home = BottomNBits(grow_home, old_shift);
  assert(old_home + (size_t{1} << old_shift) == grow_home);

  // Wait here to ensure any Grow operations that would directly feed into
  // this one are finished, though the full waiting actually completes in
  // acquiring the rewrite lock for old_home in SplitForGrow. Here we ensure
  // the expected shift amount has been reached, and there we ensure the
  // chain rewrite lock has been released.
  size_t old_old_home = BottomNBits(grow_home, old_shift - 1);
  for (;;) {
    NextWithShift old_old_head =
        array_[old_old_home].head_next_with_shift.Load();
    if (old_old_head.GetShift() >= old_shift) {
      if (!old_old_head.IsLockedNotEnd()) {
        break;
      }
    }
    // NOTE: one of the few yield-wait loops, which is rare enough in practice
    // for its performance to be insignificant.
    yield_count_.FetchAddRelaxed(1);
    std::this_thread::yield();
  }

  // FIFOCache: keys whose home was old_home now have one of the two. Their
  // tags are in old_home's group of the ghost, so grow_home's group, which
  // is another once the table has more than one, gets the live ones it
  // lacks (tags match only their own keys), before length_info_ lets
  // lookups use grow_home. An eviction that found old_home's group before
  // the update and writes its tag after the merge leaves it where the key
  // may no longer look, so the key's next insert can miss the ghost, as
  // can the keys of a group split while a SetCapacity restamps the ghost,
  // when there is no merge.
  if (grow_home >= kGhostWays) {
    GhostMerge(old_home & ~(kGhostWays - 1), grow_home & ~(kGhostWays - 1));
  }
  TEST_SYNC_POINT("FIFOCache::Grow:Merged");

  // Do the dirty work of splitting the chain, including updating heads and
  // chain nexts for new shift amounts.
  SplitForGrow(grow_home, old_home, old_shift);

  // length_info_ can be updated any time after the new shift amount is
  // published to both heads, potentially before the end of SplitForGrow.
  // But we also can't update length_info_ until the previous Grow operation
  // (with grow_home := this grow_home - 1) has published the new shift amount
  // to both of its heads. However, we don't want to artificially wait here
  // on that Grow that is otherwise irrelevant.
  //
  // We could have each Grow operation advance length_info_ here as far as it
  // can without waiting, by checking for updated shift on the corresponding
  // old home and also stopping at an empty head value for possible grow_home.
  // However, this could increase CPU cache line sharing and in 1/64 cases
  // bring in an extra page from our mmap.
  //
  // Instead, part of the strategy is delegated to DoInsert():
  // * Here we try to bring length_info_ up to date with this grow_home as
  // much as we can without waiting. It will fall short if a previous Grow
  // is still between reserving the grow slot and making the first big step
  // to publish the new shift amount.
  // * To avoid length_info_ being perpetually out-of-date (for a small number
  // of heads) after our last Grow, we do the same when Insert has to "fall
  // forward" due to length_info_ being out-of-date.
  CatchUpLengthInfoNoWait(grow_home);

  // See usage in DoInsert()
  state.likely_empty_slot = grow_home;

  // Success
  return true;
}

// See call in Grow()
void AutoHyperClockTable::CatchUpLengthInfoNoWait(
    size_t known_usable_grow_home) {
  uint64_t current_length_info = length_info_.Load();
  size_t published_usable_size = LengthInfoToUsedLength(current_length_info);
  while (published_usable_size <= known_usable_grow_home) {
    // For when published_usable_size was grow_home
    size_t next_usable_size = published_usable_size + 1;
    uint64_t next_length_info = UsedLengthToLengthInfo(next_usable_size);

    // known_usable_grow_home is known to be ready for Lookup/Insert with
    // the new shift amount, but between that and published usable size, we
    // need to check.
    if (published_usable_size < known_usable_grow_home) {
      int old_shift = FloorLog2(next_usable_size - 1);
      size_t old_home = BottomNBits(published_usable_size, old_shift);
      int shift = array_[old_home].head_next_with_shift.Load().GetShift();
      if (shift <= old_shift) {
        // Not ready
        break;
      }
    }
    // CAS update length_info_. This only moves in one direction, so if CAS
    // fails, someone else made progress like we are trying, and we can just
    // pick up the new value and keep going as appropriate.
    if (length_info_.CasStrong(current_length_info, next_length_info)) {
      current_length_info = next_length_info;
      // FIFOCache: the slot was charged when Grow took it.
    }
    published_usable_size = LengthInfoToUsedLength(current_length_info);
  }

  // After updating lengh_info_ we can update occupancy_limit_,
  // allowing for later operations to update it before us.
  // Note: there is no AcqRelAtomic max operation, so we have to use a CAS loop
  size_t old_occupancy_limit = occupancy_limit_.LoadRelaxed();
  size_t new_occupancy_limit = CalcOccupancyLimit(published_usable_size);
  while (old_occupancy_limit < new_occupancy_limit) {
    if (occupancy_limit_.CasWeakRelaxed(old_occupancy_limit,
                                        new_occupancy_limit)) {
      break;
    }
  }
}

void AutoHyperClockTable::SplitForGrow(size_t grow_home, size_t old_home,
                                       int old_shift) {
  int new_shift = old_shift + 1;
  HandleImpl* const arr = array_.Get();

  // We implement a somewhat complicated splitting algorithm to ensure that
  // entries are always wait-free visible to Lookup, without Lookup needing
  // to double-check length_info_ to ensure every potentially relevant
  // existing entry is seen. This works step-by-step, carefully sharing
  // unmigrated parts of the chain between the source chain and the new
  // destination chain. This means that Lookup might see a partially migrated
  // chain so has to take that into consideration when checking that it hasn't
  // "jumped off" its intended chain (due to a parallel modification to an
  // "under (de)construction" entry that was found on the chain but has
  // been reassigned).
  //
  // We use a "rewrite lock" on the source and desination chains to exclude
  // removals from those, and we have a prior waiting step that ensures any Grow
  // operations feeding into this one have completed. But this process does have
  // to gracefully handle concurrent insertions to the head of the source chain,
  // and once marked ready, the destination chain.
  //
  // With those considerations, the migration starts with one "big step,"
  // potentially with retries to deal with insertions in parallel. Part of the
  // big step is to mark the two chain heads as updated with the new shift
  // amount, which redirects Lookups to the appropriate new chain.
  //
  // After that big step that updates the heads, the rewrite lock makes it
  // relatively easy to deal with the rest of the migration. Big
  // simplifications come from being able to read the hashed_key of each
  // entry on the chain without needing to hold a read reference, and
  // from never "jumping our to another chain." Concurrent insertions only
  // happen at the chain head, which is outside of what is left to migrate.
  //
  // A series of smaller steps finishes splitting apart the existing chain into
  // two distinct chains, followed by some steps to fully commit the result.
  //
  // Except for trivial cases in which all entries (or remaining entries)
  // on the input chain go to one output chain, there is an important invariant
  // after each step of migration, including after the initial "big step":
  // For each output chain, the "zero chain" (new hash bit is zero) and the
  // "one chain" (new hash bit is one) we have a "frontier" entry marking the
  // boundary between what has been migrated and what has not. One of the
  // frontiers is along the old chain after the other, and all entries between
  // them are for the same target chain as the earlier frontier. Thus, the
  // chains share linked list tails starting at the latter frontier. All
  // pointers from the new head locations to the frontier entries are marked
  // with the new shift amount, while all pointers after the frontiers use the
  // old shift amount.
  //
  // And after each step there is a strengthening step to reach a stronger
  // invariant: the frontier earlier in the original chain is advanced to be
  // immediately before the other frontier.
  //
  // Consider this original input chain,
  //
  // OldHome  -Old-> A0 -Old-> B0 -Old-> A1 -Old-> C0 -Old-> OldHome(End)
  // GrowHome (empty)
  //
  // == BIG STEP ==
  // The initial big step finds the first entry that will be on the each
  // output chain (in this case A0 and A1). We use brackets ([]) to mark them
  // as our prospective frontiers.
  //
  // OldHome  -Old-> [A0] -Old-> B0 -Old-> [A1] -Old-> C0 -Old-> OldHome(End)
  // GrowHome (empty)
  //
  // Next we speculatively update grow_home head to point to the first entry for
  // the one chain. This will not be used by Lookup until the head at old_home
  // uses the new shift amount.
  //
  // OldHome  -Old-> [A0] -Old-> B0 -Old-> [A1] -Old-> C0 -Old-> OldHome(End)
  // GrowHome --------------New------------/
  //
  // Observe that if Lookup were to use the new head at GrowHome, it would be
  // able to find all relevant entries. Finishing the initial big step
  // requires a CAS (compare_exchange) of the OldHome head because there
  // might have been parallel insertions there, in which case we roll back
  // and try again. (We might need to point GrowHome head differently.)
  //
  // OldHome  -New-> [A0] -Old-> B0 -Old-> [A1] -Old-> C0 -Old-> OldHome(End)
  // GrowHome --------------New------------/
  //
  // Upgrading the OldHome head pointer with the new shift amount, with a
  // compare_exchange, completes the initial big step, with [A0] as zero
  // chain frontier and [A1] as one chain frontier. Links before the frontiers
  // use the new shift amount and links after use the old shift amount.
  // == END BIG STEP==
  // == STRENGTHENING ==
  // Zero chain frontier is advanced to [B0] (immediately before other
  // frontier) by updating pointers with new shift amounts.
  //
  // OldHome  -New-> A0 -New-> [B0] -Old-> [A1] -Old-> C0 -Old-> OldHome(End)
  // GrowHome -------------New-----------/
  //
  // == END STRENGTHENING ==
  // == SMALL STEP #1 ==
  // From the strong invariant state, we need to find the next entry for
  // the new chain with the earlier frontier. In this case, we need to find
  // the next entry for the zero chain that comes after [B0], which in this
  // case is C0. This will be our next zero chain frontier, at least under
  // the weak invariant. To get there, we simply update the link between
  // the current two frontiers to skip over the entries irreleveant to the
  // ealier frontier chain. In this case, the zero chain skips over A1. As a
  // result, he other chain is now the "earlier."
  //
  // OldHome  -New-> A0 -New-> B0 -New-> [C0] -Old-> OldHome(End)
  // GrowHome -New-> [A1] ------Old-----/
  //
  // == END SMALL STEP #1 ==
  //
  // Repeating the cycle and end handling is not as interesting.

  // Acquire rewrite lock on zero chain (if it's non-empty)
  ChainRewriteLock zero_head_lock(&arr[old_home], yield_count_);

  // Used for locking the one chain below
  NextWithShift saved_one_head;
  // One head has not been written to
  assert(arr[grow_home].head_next_with_shift.Load() ==
         HandleImpl::kUnusedMarker);

  // old_home will also the head of the new "zero chain" -- all entries in the
  // "from" chain whose next hash bit is 0. grow_home will be head of the new
  // "one chain".

  // For these, SIZE_MAX is like nullptr (unknown)
  size_t zero_chain_frontier = SIZE_MAX;
  size_t one_chain_frontier = SIZE_MAX;
  size_t cur = SIZE_MAX;

  // Set to 0 (zero chain frontier earlier), 1 (one chain), or -1 (unknown)
  int chain_frontier_first = -1;

  // Might need to retry initial update of heads
  for (int i = 0;; ++i) {
    CHECK_TOO_MANY_ITERATIONS(i);
    assert(zero_chain_frontier == SIZE_MAX);
    assert(one_chain_frontier == SIZE_MAX);
    assert(cur == SIZE_MAX);
    assert(chain_frontier_first == -1);

    NextWithShift next_with_shift = zero_head_lock.GetSavedHead();

    // Find a single representative for each target chain, or scan the whole
    // chain if some target chain has no representative.
    for (;; ++i) {
      CHECK_TOO_MANY_ITERATIONS(i);

      // Loop invariants
      assert((chain_frontier_first < 0) == (zero_chain_frontier == SIZE_MAX &&
                                            one_chain_frontier == SIZE_MAX));
      assert((cur == SIZE_MAX) == (zero_chain_frontier == SIZE_MAX &&
                                   one_chain_frontier == SIZE_MAX));

      assert(next_with_shift.GetShift() == old_shift);

      // Check for end of original chain
      if (next_with_shift.IsEnd()) {
        cur = SIZE_MAX;
        break;
      }

      // next_with_shift is not End
      cur = next_with_shift.GetNext();

      if (BottomNBits(arr[cur].hashed_key[1], new_shift) == old_home) {
        // Entry for zero chain
        if (zero_chain_frontier == SIZE_MAX) {
          zero_chain_frontier = cur;
          if (one_chain_frontier != SIZE_MAX) {
            // Ready to update heads
            break;
          }
          // Nothing yet for one chain
          chain_frontier_first = 0;
        }
      } else {
        assert(BottomNBits(arr[cur].hashed_key[1], new_shift) == grow_home);
        // Entry for one chain
        if (one_chain_frontier == SIZE_MAX) {
          one_chain_frontier = cur;
          if (zero_chain_frontier != SIZE_MAX) {
            // Ready to update heads
            break;
          }
          // Nothing yet for zero chain
          chain_frontier_first = 1;
        }
      }

      next_with_shift = arr[cur].chain_next_with_shift.Load();
    }

    // Try to update heads for initial migration info
    // We only reached the end of the migrate-from chain already if one of the
    // target chains will be empty.
    assert((cur == SIZE_MAX) ==
           (zero_chain_frontier == SIZE_MAX || one_chain_frontier == SIZE_MAX));
    assert((chain_frontier_first < 0) ==
           (zero_chain_frontier == SIZE_MAX && one_chain_frontier == SIZE_MAX));

    // Always update one chain's head first (safe), and mark it as locked
    saved_one_head = one_chain_frontier != SIZE_MAX
                         ? NextWithShift::Make(one_chain_frontier, new_shift)
                         : NextWithShift::MakeEnd(grow_home, new_shift);
    saved_one_head.Set<NextWithShift::LockedFlag>(true);
    arr[grow_home].head_next_with_shift.Store(saved_one_head);

    // Make sure length_info_ hasn't been updated too early, as we're about
    // to make the change that makes it safe to update (e.g. in DoInsert())
    assert(LengthInfoToUsedLength(length_info_.Load()) <= grow_home);

    // Try to set zero's head.
    if (zero_head_lock.CasUpdate(
            zero_chain_frontier != SIZE_MAX
                ? NextWithShift::Make(zero_chain_frontier, new_shift)
                : NextWithShift::MakeEnd(old_home, new_shift),
            yield_count_)) {
      // Both heads successfully updated to new shift
      break;
    } else {
      // Concurrent insertion. This should not happen too many times.
      CHECK_TOO_MANY_ITERATIONS(i);
      // The easiest solution is to restart.
      zero_chain_frontier = SIZE_MAX;
      one_chain_frontier = SIZE_MAX;
      cur = SIZE_MAX;
      chain_frontier_first = -1;
      continue;
    }
  }

  // Create an RAII wrapper for the one chain rewrite lock we are already
  // holding (if was not end) and is now "published" after successful CAS on
  // zero chain head.
  ChainRewriteLock one_head_lock(&arr[grow_home], yield_count_, saved_one_head);

  // Except for trivial cases, we have something like
  // AHome -New-> [A0] -Old-> [B0] -Old-> [C0] \                        |
  // BHome --------------------New------------> [A1] -Old-> ...
  // And we need to upgrade as much as we can on the "first" chain
  // (the one eventually pointing to the other's frontier). This will
  // also finish off any case in which one of the target chains will be empty.
  if (chain_frontier_first >= 0) {
    size_t& first_frontier = chain_frontier_first == 0
                                 ? /*&*/ zero_chain_frontier
                                 : /*&*/ one_chain_frontier;
    size_t& other_frontier = chain_frontier_first != 0
                                 ? /*&*/ zero_chain_frontier
                                 : /*&*/ one_chain_frontier;
    NextWithShift stop_before_or_new_tail =
        other_frontier != SIZE_MAX
            ? /*stop before*/ NextWithShift::Make(other_frontier, old_shift)
            : /*new tail*/ NextWithShift::MakeEnd(
                  chain_frontier_first == 0 ? old_home : grow_home, new_shift);
    UpgradeShiftsOnRange(arr, first_frontier, stop_before_or_new_tail,
                         old_shift, new_shift);
  }

  if (zero_chain_frontier == SIZE_MAX) {
    // Already finished migrating
    assert(one_chain_frontier == SIZE_MAX);
    assert(cur == SIZE_MAX);
  } else {
    // Still need to migrate between two target chains
    for (int i = 0;; ++i) {
      CHECK_TOO_MANY_ITERATIONS(i);
      // Overall loop invariants
      assert(zero_chain_frontier != SIZE_MAX);
      assert(one_chain_frontier != SIZE_MAX);
      assert(cur != SIZE_MAX);
      assert(chain_frontier_first >= 0);
      size_t& first_frontier = chain_frontier_first == 0
                                   ? /*&*/ zero_chain_frontier
                                   : /*&*/ one_chain_frontier;
      size_t& other_frontier = chain_frontier_first != 0
                                   ? /*&*/ zero_chain_frontier
                                   : /*&*/ one_chain_frontier;
      assert(cur != first_frontier);
      assert(arr[first_frontier].chain_next_with_shift.Load().GetNext() ==
             other_frontier);

      NextWithShift next_with_shift = arr[cur].chain_next_with_shift.Load();

      // Check for end of original chain
      if (next_with_shift.IsEnd()) {
        // Can set upgraded tail on first chain
        NextWithShift first_new_tail = NextWithShift::MakeEnd(
            chain_frontier_first == 0 ? old_home : grow_home, new_shift);
        arr[first_frontier].chain_next_with_shift.Store(first_new_tail);
        // And upgrade remainder of other chain
        NextWithShift other_new_tail = NextWithShift::MakeEnd(
            chain_frontier_first != 0 ? old_home : grow_home, new_shift);
        UpgradeShiftsOnRange(arr, other_frontier, other_new_tail, old_shift,
                             new_shift);
        assert(other_frontier == SIZE_MAX);  // Finished
        break;
      }

      // next_with_shift is not End
      cur = next_with_shift.GetNext();

      int target_chain;
      if (BottomNBits(arr[cur].hashed_key[1], new_shift) == old_home) {
        // Entry for zero chain
        target_chain = 0;
      } else {
        assert(BottomNBits(arr[cur].hashed_key[1], new_shift) == grow_home);
        // Entry for one chain
        target_chain = 1;
      }
      if (target_chain == chain_frontier_first) {
        // Found next entry to skip to on the first chain
        NextWithShift skip_to = NextWithShift::Make(cur, new_shift);
        arr[first_frontier].chain_next_with_shift.Store(skip_to);
        first_frontier = cur;
        // Upgrade other chain up to entry before that one
        UpgradeShiftsOnRange(arr, other_frontier, next_with_shift, old_shift,
                             new_shift);
        // Swap which is marked as first
        chain_frontier_first = 1 - chain_frontier_first;
        TEST_SYNC_POINT("FIFOCache::SplitForGrow:Skipped");
      } else {
        // Nothing to do yet, as we need to keep old generation pointers in
        // place for lookups
      }
    }
  }
}

// Variant of PurgeImplLocked: Removes all "under (de) construction" entries
// from a chain where already holding a rewrite lock
using PurgeLockedOpData = void;
// Variant of PurgeImplLocked: Clock-updates all entries in a chain, in
// addition to functionality of PurgeLocked, where already holding a rewrite
// lock. (Caller finalizes eviction on entries added to the autovector, in part
// so that we don't hold the rewrite lock while doing potentially expensive
// callback and allocator free.)
using ClockUpdateChainLockedOpData =
    autovector<AutoHyperClockTable::HandleImpl*>;

template <class OpData>
void AutoHyperClockTable::PurgeImplLocked(OpData* op_data,
                                          ChainRewriteLock& rewrite_lock,
                                          size_t home,
                                          BaseClockTable::EvictionData* data) {
  constexpr bool kIsPurge = std::is_same_v<OpData, PurgeLockedOpData>;
  constexpr bool kIsClockUpdateChain =
      std::is_same_v<OpData, ClockUpdateChainLockedOpData>;

  // Exactly one op specified
  static_assert(kIsPurge + kIsClockUpdateChain == 1);

  HandleImpl* const arr = array_.Get();

  NextWithShift next_with_shift = rewrite_lock.GetSavedHead();
  assert(!next_with_shift.IsEnd());
  int home_shift = next_with_shift.GetShift();
  (void)home;
  (void)home_shift;
  size_t next = next_with_shift.GetNext();
  assert(next < array_.Count());
  HandleImpl* h = &arr[next];
  HandleImpl* prev_to_keep = nullptr;
#ifndef NDEBUG
  NextWithShift prev_to_keep_next_with_shift{};
#endif
  // Whether there are entries between h and prev_to_keep that should be
  // purged from the chain.
  bool pending_purge = false;

  // Walk the chain, and stitch together any entries that are still
  // "shareable," possibly after clock update. prev_to_keep tells us where
  // the last "stitch back to" location is (nullptr => head).
  for (size_t i = 0;; ++i) {
    CHECK_TOO_MANY_ITERATIONS(i);

    bool purgeable = false;
    // In last iteration, h will be nullptr, to stitch together the tail of
    // the chain.
    if (h) {
      // NOTE: holding a rewrite lock on the chain prevents any "under
      // (de)construction" entries in the chain from being marked empty, which
      // allows us to access the hashed_keys without holding a read ref.
      assert(home == BottomNBits(h->hashed_key[1], home_shift));
      if constexpr (kIsClockUpdateChain) {
        RelaxedAtomic<uint32_t>& word = probation_words_[h - arr];
        const uint32_t w = word.LoadRelaxed();
        if (w & kInProbation) {
          // FIFOCache: the resident clock skips probation entries, and
          // records whether they are pinned. A pinned one counts against
          // the effort cap, as any pinned entry the clock passes does in
          // HyperClockCache.
          const SlotMeta meta = h->meta.Load();
          if (!meta.IsShareable()) {
            assert(!meta.IsEmpty());
            purgeable = true;
          } else {
            const bool pinned =
                meta.GetAcquireCounter() != meta.GetReleaseCounter();
            data->seen_pinned_count += pinned ? 1 : 0;
            SweepProbationEntry(*h, w, pinned);
          }
        } else {
          if (w & kCountsInsertRef) {
            DropInsertRefCount(*h, word, w);
          }
          if (ClockUpdate(*h, data, &purgeable)) {
            // Remember for finishing eviction
            op_data->push_back(h);
            // Entries for eviction become purgeable
            purgeable = true;
            assert(h->meta.Load().IsUnderConstruction());
          }
        }
      } else {
        (void)op_data;
        (void)data;
        purgeable = !h->meta.Load().IsShareable();
      }
    }

    if (purgeable) {
      assert(h->meta.Load().IsUnderConstruction());
      pending_purge = true;
    } else if (pending_purge) {
      if (prev_to_keep) {
        // Update chain next to skip purgeable entries
        assert(prev_to_keep->chain_next_with_shift.Load() ==
               prev_to_keep_next_with_shift);
        prev_to_keep->chain_next_with_shift.Store(next_with_shift);
      } else if (rewrite_lock.CasUpdate(next_with_shift, yield_count_)) {
        // Managed to update head without any parallel insertions
      } else {
        // Parallel insertion must have interfered. Need to do a purge
        // from updated head to here. Since we have no prev_to_keep, there's
        // no risk of duplicate clock updates to entries. Any entries already
        // updated must have been evicted (purgeable) and it's OK to clock
        // update any new entries just inserted in parallel.
        // Can simply restart (GetSavedHead() already updated from CAS failure).
        next_with_shift = rewrite_lock.GetSavedHead();
        assert(!next_with_shift.IsEnd());
        next = next_with_shift.GetNext();
        assert(next < array_.Count());
        h = &arr[next];
        pending_purge = false;
        assert(prev_to_keep == nullptr);
        assert(next_with_shift.GetShift() == home_shift);
        continue;
      }
      pending_purge = false;
      prev_to_keep = h;
    } else {
      prev_to_keep = h;
    }

    if (h == nullptr) {
      // Reached end of the chain
      return;
    }

    // Read chain pointer
    next_with_shift = h->chain_next_with_shift.Load();
#ifndef NDEBUG
    if (prev_to_keep == h) {
      prev_to_keep_next_with_shift = next_with_shift;
    }
#endif

    assert(next_with_shift.GetShift() == home_shift);

    // Check for end marker
    if (next_with_shift.IsEnd()) {
      h = nullptr;
    } else {
      next = next_with_shift.GetNext();
      assert(next < array_.Count());
      h = &arr[next];
      assert(h != prev_to_keep);
    }
  }
}

// Variant of PurgeImpl: Removes all "under (de) construction" entries in a
// chain, such that any entry with the given key must have been purged.
using PurgeOpData = const UniqueId64x2;
// Variant of PurgeImpl: Clock-updates all entries in a chain, in addition to
// purging as appropriate. (Caller finalizes eviction on entries added to the
// autovector, in part so that we don't hold the rewrite lock while doing
// potentially expensive callback and allocator free.)
using ClockUpdateChainOpData = ClockUpdateChainLockedOpData;

template <class OpData>
void AutoHyperClockTable::PurgeImpl(OpData* op_data, size_t home,
                                    BaseClockTable::EvictionData* data) {
  // Early efforts to make AutoHCC fully wait-free ran into too many problems
  // that needed obscure and potentially inefficient work-arounds to have a
  // chance at working.
  //
  // The implementation settled on "essentially wait-free" which can be
  // achieved by locking at the level of each probing chain and only for
  // operations that might remove entries from the chain. Because parallel
  // clock updates and Grow operations are ordered, contention is very rare.
  // However, parallel insertions at any chain head have to be accommodated
  // to keep them wait-free.
  //
  // This function implements Purge and ClockUpdateChain functions (see above
  // OpData type definitions) as part of higher-level operations. This function
  // ensures the correct chain is (eventually) covered and handles rewrite
  // locking the chain. PurgeImplLocked has lower level details.
  //
  // In general, these operations and Grow are kept simpler by allowing eager
  // purging of under (de-)construction entries. For example, an Erase
  // operation might find that another thread has purged the entry from the
  // chain by the time its own purge operation acquires the rewrite lock and
  // proceeds. This is OK, and potentially reduces the number of lock/unlock
  // cycles because empty chains are not rewrite-lockable.

  constexpr bool kIsPurge = std::is_same_v<OpData, PurgeOpData>;
  constexpr bool kIsClockUpdateChain =
      std::is_same_v<OpData, ClockUpdateChainOpData>;

  // Exactly one op specified
  static_assert(kIsPurge + kIsClockUpdateChain == 1);

  int home_shift = 0;
  if constexpr (kIsPurge) {
    // Purge callers leave home unspecified, to be determined from key
    assert(home == SIZE_MAX);
    GetHomeIndexAndShift(length_info_.Load(), (*op_data)[1], &home,
                         &home_shift);
    assert(home_shift > 0);
  } else {
    assert(kIsClockUpdateChain);
    // Evict callers must specify home
    assert(home < SIZE_MAX);
  }

  HandleImpl* const arr = array_.Get();

  // Acquire the RAII rewrite lock (if not an empty chain)
  ChainRewriteLock rewrite_lock(&arr[home], yield_count_);

  if constexpr (kIsPurge) {
    // Ensure we are at the correct home for the shift in effect for the
    // chain head.
    for (;;) {
      int shift = rewrite_lock.GetSavedHead().GetShift();

      if (shift > home_shift) {
        // Found a newer shift at candidate head, which must apply to us.
        // Newer shift might not yet be reflected in length_info_ (an atomicity
        // gap in Grow), so operate as if it is. Note that other insertions
        // could happen using this shift before length_info_ is updated, and
        // it's possible (though unlikely) that multiple generations of Grow
        // have occurred. If shift is more than one generation ahead of
        // home_shift, it's possible that not all descendent homes have
        // reached the `shift` generation. Thus, we need to advance only one
        // shift at a time looking for a home+head with a matching shift
        // amount.
        home_shift++;
        home = GetHomeIndex((*op_data)[1], home_shift);
        rewrite_lock.Reset(&arr[home], yield_count_);
        continue;
      } else {
        assert(shift == home_shift);
      }
      break;
    }
  }

  // If the chain is empty, nothing to do
  if (!rewrite_lock.IsEnd()) {
    if constexpr (kIsPurge) {
      PurgeLockedOpData* locked_op_data{};
      PurgeImplLocked(locked_op_data, rewrite_lock, home, data);
    } else {
      PurgeImplLocked(op_data, rewrite_lock, home, data);
    }
  }
}

AutoHyperClockTable::HandleImpl* AutoHyperClockTable::DoInsert(
    const ClockHandleBasicData& proto, uint32_t initial_countdown,
    bool take_ref, InsertState& state) {
  size_t home;
  int orig_home_shift;
  GetHomeIndexAndShift(state.saved_length_info, proto.hashed_key[1], &home,
                       &orig_home_shift);
  HandleImpl* const arr = array_.Get();

  // FIFOCache: HyperClockCache skips the insert if the home slot, or another
  // it probes, holds an entry with the key. Here the probes don't check keys
  // (see BeginSlotInsert): the entry is linked and then replaces any other
  // with its key.
  //
  // The search for an available slot starts at the home slot. As the table
  // initially grows to size, few entries will be in the same cache line as the
  // chain head. However, churn in the cache relatively quickly improves the
  // proportion of entries sharing that cache line with the chain head. Data:
  //
  // Initial population only: (cache_bench with -ops_per_thread=1)
  // Entries at home count: 29,202 (out of 129,170 entries in 94,411 chains)
  // Approximate average cache lines read to find an existing entry:
  //           129.2 / 94.4 [without the heads]
  // + (94.4 - 29.2) / 94.4 [the heads not included with entries]
  // = 2.06 cache lines
  //
  // After 10 million ops: (-threads=10 -ops_per_thread=100000)
  // Entries at home count: 67,556 (out of 129,359 entries in 94,756 chains)
  // That's a majority of entries and more than 2/3rds of chains.
  // Approximate average cache lines read to find an existing entry:
  // = 1.65 cache lines

  // Even if we aren't saving a ref to this entry (take_ref == false), we need
  // to keep a reference while we are inserting the entry into a chain, so that
  // it is not erased by another thread while trying to insert it on the chain.
  constexpr bool initial_take_ref = true;
  static_assert(initial_take_ref,
                "FIFOCache: TryInsertSlot always takes the reference");

  // FIFOCache: the entry goes into the probation queue, unless Insert took
  // its key's ghost tag or it is a cache reservation, when it goes to the
  // resident queue. Its probation word records whether the caller gets a
  // reference, and whether it is a reservation.
  const uint32_t word_flags =
      (state.ghost_hit || state.reservation ? uint32_t{0} : kInProbation) |
      (take_ref ? kCountsInsertRef : uint32_t{0}) |
      (state.reservation ? kReservation : uint32_t{0});
  uint32_t word = 0;

  size_t used_length = LengthInfoToUsedLength(state.saved_length_info);
  assert(home < used_length);

  size_t idx = home;
  if (TryInsertSlot(proto, idx, initial_countdown, word_flags,
                    state.probation_precharged, &word)) {
    assert(idx == home);
    // Here we try to populate newly-opened slots in the table, but not
    // when we can add something to its home slot. This makes the structure
    // more performant more quickly on (initial) growth.
  } else if (UNLIKELY(state.likely_empty_slot > 0) &&
             TryInsertSlot(proto, state.likely_empty_slot, initial_countdown,
                           word_flags, state.probation_precharged, &word)) {
    idx = state.likely_empty_slot;
  } else {
    // We need to search for an available slot outside of the home.
    // Linear hashing provides nice resizing but does typically mean
    // that some heads (home locations) have (in expectation) twice as
    // many entries mapped to them as other heads. For example if the
    // usable length is 80, then heads 16-63 are (in expectation) twice
    // as loaded as heads 0-15 and 64-79, which are using another hash bit.
    //
    // This means that if we just use linear probing (by a small constant)
    // to find an available slot, part of the structure could easily fill up
    // and resort to linear time operations even when the overall load factor
    // is only modestly high, like 70%. Even though each slot has its own CPU
    // cache line, there appears to be a small locality benefit (e.g. TLB and
    // paging) to iterating one by one, as long as we don't afoul of the
    // linear hashing imbalance.
    //
    // In a traditional non-concurrent structure, we could keep a "free list"
    // to ensure immediate access to an available slot, but maintaining such
    // a structure could require more cross-thread coordination to ensure
    // all entries are eventually available to all threads.
    //
    // The way we solve this problem is to use unit-increment linear probing
    // with a small bound, and then fall back on big jumps to have a good
    // chance of finding a slot in an under-populated region quickly if that
    // doesn't work.
    size_t i = 0;
    constexpr size_t kMaxLinearProbe = 4;
    for (; i < kMaxLinearProbe; i++) {
      idx++;
      if (idx >= used_length) {
        idx -= used_length;
      }
      if (TryInsertSlot(proto, idx, initial_countdown, word_flags,
                        state.probation_precharged, &word)) {
        break;
      }
    }
    if (i == kMaxLinearProbe) {
      // Keep searching, but change to a search method that should quickly
      // find any under-populated region. Switching to an increment based
      // on the golden ratio helps with that, but we also inject some minor
      // variation (less than 2%, 1 in 2^6) to avoid clustering effects on
      // this larger increment (if it were a fixed value in steady state
      // operation). Here we are primarily using upper bits of hashed_key[1]
      // while home is based on lowest bits.
      uint64_t incr_ratio = 0x9E3779B185EBCA87U + (proto.hashed_key[1] >> 6);
      size_t incr = FastRange64(incr_ratio, used_length);
      assert(incr > 0);
      size_t start = idx;
      for (;; i++) {
        idx += incr;
        if (idx >= used_length) {
          // Wrap around (faster than %)
          idx -= used_length;
        }
        if (idx == start) {
          // We have just completed a cycle that might not have covered all
          // slots. (incr and used_length could have common factors.)
          // Increment for the next cycle, which eventually ensures complete
          // iteration over the set of slots before repeating.
          idx++;
          if (idx >= used_length) {
            idx -= used_length;
          }
          start++;
          if (start >= used_length) {
            start -= used_length;
          }
          if (i >= used_length) {
            used_length = LengthInfoToUsedLength(length_info_.Load());
            if (i >= used_length * 2) {
              // Cycling back should not happen unless there is enough random
              // churn in parallel that we happen to hit each slot at a time
              // that it's occupied, which is really only feasible for small
              // structures, though with linear probing to find empty slots,
              // "small" here might be larger than for double hashing.
              assert(used_length <= 256);
              // Fall back on standalone insert in case something goes awry to
              // cause this
              return nullptr;
            }
          }
        }
        if (TryInsertSlot(proto, idx, initial_countdown, word_flags,
                          state.probation_precharged, &word)) {
          break;
        }
      }
    }
  }

  // Now insert into chain using head pointer
  NextWithShift next_with_shift;
  int home_shift = orig_home_shift;

  // Might need to retry
  for (int i = 0;; ++i) {
    CHECK_TOO_MANY_ITERATIONS(i);
    next_with_shift = arr[home].head_next_with_shift.Load();
    int shift = next_with_shift.GetShift();

    if (UNLIKELY(shift != home_shift)) {
      // NOTE: shift increases with table growth
      if (shift > home_shift) {
        // Must be grow in progress or completed since reading length_info.
        // Pull out one more hash bit. (See Lookup() for why we can't
        // safely jump to the shift that was read.)
        home_shift++;
        uint64_t hash_bit_mask = uint64_t{1} << (home_shift - 1);
        assert((home & hash_bit_mask) == 0);
        // BEGIN leftover updates to length_info_ for Grow()
        size_t grow_home = home + hash_bit_mask;
        assert(arr[grow_home].head_next_with_shift.Load() !=
               HandleImpl::kUnusedMarker);
        CatchUpLengthInfoNoWait(grow_home);
        // END leftover updates to length_info_ for Grow()
        home += proto.hashed_key[1] & hash_bit_mask;
        continue;
      } else {
        // Should not happen because length_info_ is only updated after both
        // old and new home heads are marked with new shift
        assert(false);
      }
    }

    // Values to update to
    NextWithShift head_next_with_shift = NextWithShift::Make(idx, home_shift);
    NextWithShift chain_next_with_shift = next_with_shift;

    // Preserve the locked state in head, without propagating to chain next
    // where it is meaningless (and not allowed)
    if (UNLIKELY(next_with_shift.IsLockedNotEnd())) {
      head_next_with_shift.Set<NextWithShift::LockedFlag>(true);
      chain_next_with_shift.Set<NextWithShift::LockedFlag>(false);
    }

    arr[idx].chain_next_with_shift.Store(chain_next_with_shift);
    if (arr[home].head_next_with_shift.CasWeak(next_with_shift,
                                               head_next_with_shift)) {
      // Success
      TEST_SYNC_POINT("FIFOCache::DoInsert:Linked");
      // FIFOCache: a ghost return counts once its entry is in the table.
      if (state.ghost_hit) {
        ghost_returns_.FetchAddRelaxed(1);
      }
      // FIFOCache: replaces any other entry with the key, while our
      // reference keeps this one in the table.
      ReplacedEntries replaced;
      state.replaced = EraseReplaced(proto, arr + idx, &replaced);
      if (state.replaced) {
        TEST_SYNC_POINT("FIFOCache::DoInsert:BeforeInherit");
        word = TakeReplacedPlace(idx, word, replaced, proto.GetTotalCharge());
      }
      if (!take_ref) {
        // Frees it if it was erased meanwhile and this was the last
        // reference.
        Release(arr + idx, /*useful=*/false, /*erase_if_last_ref=*/false);
      }
      // FIFOCache: queue it once it is linked and our reference is gone, so
      // a probation step sees it whole and not pinned by us. If it was
      // removed meanwhile, the cell is stale and the step skips it.
      if (word & kInProbation) {
        TEST_SYNC_POINT("FIFOCache::DoInsert:BeforeProbationPush");
        if (!ProbationPush(idx, word, state) &&
            LeaveProbation(idx, word, proto.GetTotalCharge())) {
          probation_full_.FetchAddRelaxed(1);
        }
      }
      return arr + idx;
    }
  }
}

void AutoHyperClockTable::EvictOverCapacity(size_t up_to) {
  // Only the deficit at the start, as inserts racing this one evict for
  // their own charges. Each Evict call gives up at the effort cap, so call
  // it again while calls free entries or find pins, until the pins found
  // since anything was freed could have covered every entry twice.
  const size_t usage = usage_.LoadRelaxed();
  const size_t capacity = capacity_.LoadRelaxed();
  size_t deficit = usage > capacity ? std::min(up_to, usage - capacity) : 0;
  const size_t pins_limit = 2 * occupancy_.LoadRelaxed();
  size_t pins_since_freed = 0;
  bool effort_cap_reached = false;
  while (deficit > 0) {
    InsertState state;
    StartInsert(state);
    EvictionData data;
    Evict(deficit, /*need_entry=*/false, state, &data);
    effort_cap_reached |= data.effort_cap_reached;
    occupancy_.FetchSub(data.freed_count);
    usage_.FetchSubRelaxed(data.freed_charge);
    if (data.freed_charge > 0) {
      deficit -= std::min(deficit, data.freed_charge);
      pins_since_freed = 0;
    } else {
      // Freeing entries of no charge takes nothing off the deficit, so it
      // doesn't reset the count, or entries of no charge arriving meanwhile
      // could keep this going.
      pins_since_freed += data.seen_pinned_count;
      if (data.seen_pinned_count == 0 || pins_since_freed > pins_limit) {
        break;
      }
    }
    // And for a ring that a rotation in this call charged, as an insert
    // evicts for one (see EvictForChargedRing), as far as usage is over
    // capacity: a racing erase may already have made room for it.
    if (UNLIKELY(state.ring_charge > 0)) {
      const size_t now = usage_.LoadRelaxed();
      deficit = std::min(deficit + state.ring_charge,
                         now > capacity ? now - capacity : 0);
    }
  }
  // One effort cap stop for the cut, and only if it ends over capacity:
  // the calls before the last that gave up at the cap were retried.
  if (deficit > 0 && effort_cap_reached) {
    eviction_effort_exceeded_count_.FetchAddRelaxed(1);
  }
}

AutoHyperClockTable::HandleImpl* AutoHyperClockTable::Lookup(
    const UniqueId64x2& hashed_key, const Slice* key, const HandleImpl* skip) {
  // FIFOCache: each reference this drops goes through Release (except
  // MatchAndRef's on a mismatch), which frees an erased entry if that
  // reference was its last, taking the chain's rewrite lock as any release
  // of an erased entry does. Left for eviction, which a cache under its
  // capacity and occupancy limit doesn't run, the erased copies of a key
  // often replaced or erased would pile up in its chain.
  //
  // Lookups are wait-free with low occurrence of retries, back-tracking,
  // and fallback. We do not have the benefit of holding a rewrite lock on
  // the chain so must be prepared for many kinds of mayhem, most notably
  // "falling off our chain" where a slot that Lookup has identified but
  // has not read-referenced is removed from one chain and inserted into
  // another. The full algorithm uses the following mitigation strategies to
  // ensure every relevant entry inserted before this Lookup, and not yet
  // evicted, is seen by Lookup, without excessive backtracking etc.:
  // * Keep a known good read ref in the chain for "island hopping." When
  // we observe that a concurrent write takes us off to another chain, we
  // only need to fall back to our last known good read ref (most recent
  // entry on the chain that is not "under construction," which is a transient
  // state). We don't want to compound the CPU toil of a long chain with
  // operations that might need to retry from scratch, with probability
  // in proportion to chain length.
  // * Only detect a chain is potentially incomplete because of a Grow in
  // progress by looking at shift in the next pointer tags (rather than
  // re-checking length_info_).
  // * SplitForGrow, Insert, and PurgeImplLocked ensure that there are no
  // transient states that might cause this full Lookup algorithm to skip over
  // live entries.

  // Reading length_info_ is not strictly required for Lookup, if we were
  // to increment shift sizes until we see a shift size match on the
  // relevant head pointer. Thus, reading with relaxed memory order gives
  // us a safe and almost always up-to-date jump into finding the correct
  // home and head.
  size_t home;
  int home_shift;
  GetHomeIndexAndShift(length_info_.LoadRelaxed(), hashed_key[1], &home,
                       &home_shift);
  assert(home_shift > 0);

  // The full Lookup algorithm however is not great for hot path efficiency,
  // because of the extra careful tracking described above. Overwhelmingly,
  // we can find what we're looking for with a naive linked list traversal
  // of the chain. Even if we "fall off our chain" to another, we don't
  // violate memory safety. We just won't match the key we're looking for.
  // And we would eventually reach an end state, possibly even experiencing a
  // cycle as an entry is freed and reused during our traversal (though at
  // any point in time the structure doesn't have cycles).
  //
  // So for hot path efficiency, we start with a naive Lookup attempt, and
  // then fall back on full Lookup if we don't find the correct entry. To
  // cap how much we invest into the naive Lookup, we simply cap the traversal
  // length before falling back. Also, when we do fall back on full Lookup,
  // we aren't paying much penalty by starting over. Much or most of the cost
  // of Lookup is memory latency in following the chain pointers, and the
  // naive Lookup has warmed the CPU cache for these entries, using as tight
  // of a loop as possible.

  HandleImpl* const arr = array_.Get();
  NextWithShift next_with_shift = arr[home].head_next_with_shift.LoadRelaxed();
  for (size_t i = 0; !next_with_shift.IsEnd() && i < 10; ++i) {
    HandleImpl* h = &arr[next_with_shift.GetNext()];
    // Attempt cheap key match without acquiring a read ref. This could give a
    // false positive, which is re-checked after acquiring read ref, or false
    // negative, which is re-checked in the full Lookup. Also, this is a
    // technical UB data race according to TSAN, but we don't need to read
    // a "correct" value here for correct overall behavior.
#ifdef __SANITIZE_THREAD__
    bool probably_equal = Random::GetTLSInstance()->OneIn(2);
#else
    bool probably_equal = h->hashed_key == hashed_key;
#endif
    // FIFOCache: or skip.
    if (probably_equal && h != skip) {
      // Increment acquire counter for definitive check
      auto add_acquire = AcquireCounter::PlusTransformPromiseNoOverflow(1);
      SlotMeta old_meta, new_meta;
      h->meta.Apply(add_acquire, &old_meta, &new_meta);
      // Check if it's a referencable (sharable) entry
      if (LIKELY(old_meta.IsShareable())) {
        assert(new_meta.GetRefcount() > 0);
        if (LIKELY(h->MatchesKey(hashed_key, key)) &&
            LIKELY(old_meta.IsVisible())) {
          return h;
        } else {
          Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
        }
      } else {
        // For non-sharable states, incrementing the acquire counter has no
        // effect so we don't need to undo it. Furthermore, we cannot safely
        // undo it because we did not acquire a read reference to lock the entry
        // in a Shareable state.
      }
    }

    next_with_shift = h->chain_next_with_shift.LoadRelaxed();
  }

  // If we get here, falling back on full Lookup algorithm.
  HandleImpl* h = nullptr;
  HandleImpl* read_ref_on_chain = nullptr;

  for (size_t i = 0;; ++i) {
    CHECK_TOO_MANY_ITERATIONS(i);
    // Read head or chain pointer
    next_with_shift = h ? h->chain_next_with_shift.Load()
                        : arr[home].head_next_with_shift.Load();
    int shift = next_with_shift.GetShift();

    // Make sure it's usable
    size_t effective_home = home;
    if (UNLIKELY(shift != home_shift)) {
      // We have potentially gone awry somehow, but it's possible we're just
      // hitting old data that is not yet completed Grow.
      // NOTE: shift bits goes up with table growth.
      if (shift < home_shift) {
        // To avoid waiting on Grow in progress, an old shift amount needs
        // to be processed as if we were still using it and (potentially
        // different or the same) the old home.
        // We can assert it's not too old, because each generation of Grow
        // waits on its ancestor in the previous generation.
        assert(shift + 1 == home_shift);
        effective_home = GetHomeIndex(home, shift);
      } else if (h == read_ref_on_chain) {
        assert(shift > home_shift);
        // At head or coming from an entry on our chain where we're holding
        // a read reference. Thus, we know the newer shift applies to us.
        // Newer shift might not yet be reflected in length_info_ (an atomicity
        // gap in Grow), so operate as if it is. Note that other insertions
        // could happen using this shift before length_info_ is updated, and
        // it's possible (though unlikely) that multiple generations of Grow
        // have occurred. If shift is more than one generation ahead of
        // home_shift, it's possible that not all descendent homes have
        // reached the `shift` generation. Thus, we need to advance only one
        // shift at a time looking for a home+head with a matching shift
        // amount.
        home_shift++;
        // Update home in case it has changed
        home = GetHomeIndex(hashed_key[1], home_shift);
        // This should be rare enough occurrence that it's simplest just
        // to restart (TODO: improve in some cases?)
        h = nullptr;
        if (read_ref_on_chain) {
          Release(read_ref_on_chain, /*useful=*/false,
                  /*erase_if_last_ref=*/false);
          read_ref_on_chain = nullptr;
        }
        // Didn't make progress & retry
        continue;
      } else {
        assert(shift > home_shift);
        assert(h != nullptr);
        // An "under (de)construction" entry has a new shift amount, which
        // means we have either gotten off our chain or our home shift is out
        // of date. If we revert back to saved ref, we will get updated info.
        h = read_ref_on_chain;
        // Didn't make progress & retry
        continue;
      }
    }

    // Check for end marker
    if (next_with_shift.IsEnd()) {
      // To ensure we didn't miss anything in the chain, the end marker must
      // point back to the correct home.
      if (LIKELY(next_with_shift.GetNext() == effective_home)) {
        // Complete, clean iteration of the chain, not found.
        // Clean up.
        if (read_ref_on_chain) {
          Release(read_ref_on_chain, /*useful=*/false,
                  /*erase_if_last_ref=*/false);
        }
        return nullptr;
      } else {
        // Something went awry. Revert back to a safe point (if we have it)
        h = read_ref_on_chain;
        // Didn't make progress & retry
        continue;
      }
    }

    // Follow the next and check for full key match, home match, or neither
    h = &arr[next_with_shift.GetNext()];
    bool full_match_or_unknown = false;
    if (MatchAndRef(&hashed_key, key, *h, shift, effective_home,
                    &full_match_or_unknown)) {
      // Got a read ref on next (h).
      //
      // There is a very small chance that between getting the next pointer
      // (now h) and doing MatchAndRef on it, another thread erased/evicted it
      // reinserted it into the same chain, causing us to cycle back in the
      // same chain and potentially see some entries again if we keep walking.
      // Newly-inserted entries are inserted before older ones, so we are at
      // least guaranteed not to miss anything. Here in Lookup, it's just a
      // transient, slight hiccup in performance.

      // FIFOCache: skip is kept as a read ref on the chain, like an entry
      // with the same home.
      if (full_match_or_unknown && h != skip) {
        // Full match.
        // Release old read ref on chain if applicable
        if (read_ref_on_chain) {
          // Pretend we never took the reference.
          Release(read_ref_on_chain, /*useful=*/false,
                  /*erase_if_last_ref=*/false);
        }
        // Update the hit bit
        if (eviction_callback_) {
          h->meta.ApplyRelaxed(SlotMeta::HitFlag::SetTransform());
        }
        // All done.
        return h;
      } else if (UNLIKELY(shift != home_shift) &&
                 home != BottomNBits(h->hashed_key[1], home_shift)) {
        // This chain is in a Grow operation and we've landed on an entry
        // that belongs to the wrong destination chain. We can keep going, but
        // there's a chance we'll need to backtrack back *before* this entry,
        // if the Grow finishes before this Lookup. We cannot save this entry
        // for backtracking because it might soon or already be on the wrong
        // chain.
        // NOTE: if we simply backtrack rather than continuing, we would
        // be in a wait loop (not allowed in Lookup!) until the other thread
        // finishes its Grow.
        Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
      } else {
        // Correct home location, so we are on the right chain.
        // With new usable read ref, can release old one (if applicable).
        if (read_ref_on_chain) {
          // Pretend we never took the reference.
          Release(read_ref_on_chain, /*useful=*/false,
                  /*erase_if_last_ref=*/false);
        }
        // And keep the new one.
        read_ref_on_chain = h;
      }
    } else {
      if (full_match_or_unknown) {
        // Must have been an "under construction" entry. Can safely skip it,
        // but there's a chance we'll have to backtrack later
      } else {
        // Home mismatch! Revert back to a safe point (if we have it)
        h = read_ref_on_chain;
        // Didn't make progress & retry
      }
    }
  }
}

void AutoHyperClockTable::Remove(HandleImpl* h) {
  assert(h->meta.Load().IsUnderConstruction());

  const HandleImpl& c_h = *h;
  PurgeImpl(&c_h.hashed_key);
}

bool AutoHyperClockTable::TryEraseHandle(HandleImpl* h, bool holding_ref,
                                         bool mark_invisible,
                                         bool only_if_invisible) {
  TEST_SYNC_POINT_CALLBACK("FIFOCache::TryEraseHandle:Begin", h);
  SlotMeta meta = h->meta.Load();
  assert(!holding_ref || meta.IsShareable());

  // Take ownership if no other refs, or set invisible if other refs exist (and
  // mark_invisible is set).
  SlotMeta construction_meta;
  construction_meta.SetUnderConstruction();
  do {
    if (meta.GetRefcount() != uint32_t{holding_ref}) {
      // Not last ref at some point in time during this call
      if (mark_invisible) {
        // Set invisible
        h->meta.Apply(SlotMeta::VisibleFlag::ClearTransform());
      }
      return false;
    }
    if (!meta.IsShareable()) {
      // Someone else took ownership
      return false;
    }
    if (only_if_invisible && meta.IsVisible()) {
      // FIFOCache: freed since, and the slot reused by a new entry.
      return false;
    }
    // Note that if !holding_ref, there's a small chance that we release,
    // another thread replaces this entry with another, reaches zero refs, and
    // then we end up erasing that other entry. That's an acceptable risk /
    // imprecision. (FIFOCache: only_if_invisible rules it out where the
    // entry was erased.)
  } while (!h->meta.CasWeak(meta, construction_meta));
  // Took ownership
  // TODO? Delay freeing?
  h->FreeData(allocator_);
  size_t total_charge = h->total_charge;
  if (UNLIKELY(h->IsStandalone())) {
    // Delete detached handle
    delete h;
    standalone_usage_.FetchSubRelaxed(total_charge);
  } else {
    // FIFOCache: before the slot can be reused.
    TrackRemoval(static_cast<size_t>(h - array_.Get()), total_charge);
    Remove(h);
    MarkEmpty(*h);
    occupancy_.FetchSub(1U);
  }
  usage_.FetchSubRelaxed(total_charge);
  assert(usage_.LoadRelaxed() < SIZE_MAX / 2);
  return true;
}

bool AutoHyperClockTable::Release(HandleImpl* h, bool useful,
                                  bool erase_if_last_ref) {
  // In contrast with LRUCache's Release, this function won't delete the handle
  // when the cache is above capacity and the reference is the last one. Space
  // is only freed up by Evict/PurgeImpl (called by Insert when space
  // is needed) and Erase. We do this to avoid an extra atomic read of the
  // variable usage_.

  if (erase_if_last_ref) {
    // FIFOCache: decided while this reference still holds the entry, so the
    // erase can't reach a later entry in a slot emptied and refilled after
    // the reference was dropped.
    SlotMeta meta = h->meta.Load();
    for (;;) {
      assert(meta.IsShareable());
      // No underflow
      assert(meta.GetAcquireCounter() != meta.GetReleaseCounter());
      if (meta.GetRefcount() == 1) {
        if (TryEraseHandle(h, /*holding_ref=*/true, /*mark_invisible=*/false,
                           /*only_if_invisible=*/false)) {
          return true;
        }
        // Referenced again meanwhile.
        TEST_SYNC_POINT("FIFOCache::Release:EraseRefTaken");
        meta = h->meta.Load();
        continue;
      }
      // Not the last: drops this one, unless the counts changed meanwhile.
      SlotMeta dropped = meta;
      if (useful) {
        dropped.SetReleaseCounter(meta.GetReleaseCounter() + 1);
      } else {
        dropped.SetAcquireCounter(meta.GetAcquireCounter() - 1);
      }
      if (h->meta.CasWeak(meta, dropped)) {
        if (useful) {
          CorrectNearOverflow(meta, h->meta);
        }
        return false;
      }
    }
  }

  SlotMeta old_meta;
  if (useful) {
    // Increment release counter to indicate was used
    auto add_release = ReleaseCounter::PlusTransformPromiseNoOverflow(1);
    h->meta.Apply(add_release, &old_meta);
    // Correct for possible (but rare) overflow
    CorrectNearOverflow(old_meta, h->meta);
  } else {
    // Decrement acquire counter to pretend it never happened
    auto sub_acquire = AcquireCounter::MinusTransformPromiseNoUnderflow(1);
    h->meta.Apply(sub_acquire, &old_meta);
  }

  assert(old_meta.IsShareable());
  // No underflow
  assert(old_meta.GetAcquireCounter() != old_meta.GetReleaseCounter());

  if (UNLIKELY(old_meta.IsInvisible())) {
    // FIFOCache: erased while referenced. Frees it if this was the last
    // reference, and not a new entry in the slot if another release freed
    // it first.
    TEST_SYNC_POINT("FIFOCache::Release:BeforeErase");
    return TryEraseHandle(h, /*holding_ref=*/false, /*mark_invisible=*/false,
                          /*only_if_invisible=*/true);
  } else {
    return false;
  }
}

#ifndef NDEBUG
void AutoHyperClockTable::TEST_ReleaseN(HandleImpl* h, uint32_t n) {
  if (n > 0) {
    // Do n-1 simple releases first
    TEST_ReleaseNMinus1(h, n);

    // Then the last release might be more involved
    Release(h, /*useful*/ true, /*erase_if_last_ref*/ false);
  }
}
#endif

void AutoHyperClockTable::Erase(const UniqueId64x2& hashed_key,
                                const Slice* key) {
  // Don't need to be efficient.
  // Might be one match masking another, so loop.
  while (HandleImpl* h = Lookup(hashed_key, key)) {
    bool gone = TryEraseHandle(h, /*holding_ref=*/true, /*mark_invisible=*/true,
                               /*only_if_invisible=*/false);
    if (!gone) {
      // Only marked invisible, which is ok.
      TEST_SYNC_POINT("FIFOCache::Erase:Marked");
      // Pretend we never took the reference from Lookup. FIFOCache: unless
      // the holders released theirs meanwhile, when this one is the last and
      // frees the entry.
      Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
    }
  }
}

const AutoHyperClockTable::HandleImpl* AutoHyperClockTable::FirstLinked(
    const HandleImpl* a, const HandleImpl* b,
    const UniqueId64x2& hashed_key) const {
  // As Lookup's first, naive pass: references keep both entries linked, and
  // a chain keeps its entries' order through purges and splits, so the one
  // the walk reaches first comes first, as long as the walk stays on the
  // key's chain. A head with a newer shift than length_info_ gives (a split
  // whose length isn't published yet, as Insert and Lookup find) moves the
  // walk to the key's home under that shift, one bit at a time. Past the
  // head, a link with the older shift is the tail a split in progress still
  // shares between its two chains, in order. One with a newer shift is from
  // a split of the chain since the walk read its head, and the split's new
  // links skip the entries bound for its other chain, which may be the
  // key's, so as at an end before either entry, or past kMaxChainWalk, the
  // walk can't tell.
  size_t home;
  int home_shift;
  GetHomeIndexAndShift(length_info_.Load(), hashed_key[1], &home, &home_shift);
  const HandleImpl* const arr = array_.Get();
  NextWithShift next_with_shift = arr[home].head_next_with_shift.Load();
  while (next_with_shift.GetShift() > home_shift) {
    ++home_shift;
    home = GetHomeIndex(hashed_key[1], home_shift);
    next_with_shift = arr[home].head_next_with_shift.Load();
  }
  TEST_SYNC_POINT("FIFOCache::FirstLinked:Head");
  for (size_t i = 0; i < kMaxChainWalk; ++i) {
    if (next_with_shift.IsEnd() || next_with_shift.GetShift() > home_shift) {
      return nullptr;
    }
    const HandleImpl* h = &arr[next_with_shift.GetNext()];
    if (h == a || h == b) {
      return h;
    }
    next_with_shift = h->chain_next_with_shift.Load();
  }
  return nullptr;
}

bool AutoHyperClockTable::EraseReplaced(const ClockHandleBasicData& proto,
                                        HandleImpl* keep,
                                        ReplacedEntries* replaced_entries) {
  Slice heap_key;
  const Slice* key = nullptr;
  if (proto.HasHeapKey()) {
    heap_key = proto.GetHeapKey()->key();
    key = &heap_key;
  }
  bool replaced = false;
  // Like Erase. If keep is erased meanwhile, by Erase or by a newer insert
  // replacing it, that erases the rest, which may include the newer entry.
  // An entry linked after keep is a newer insert's, which replaces keep and
  // the rest: erasing it here could erase both while the newer insert erases
  // keep, leaving an older entry visible. So h goes only if keep is found
  // ahead of it. When the walk can't tell, keep is hidden first, so a newer
  // insert finds nothing to erase and stops, and then h and any other entry
  // with the key go: the key is absent until its next insert, which is a
  // miss, never an older value.
  bool unsure = false;
  while (HandleImpl* h = Lookup(proto.hashed_key, key, keep)) {
    if (keep != nullptr && !unsure) {
      const HandleImpl* first = keep->meta.Load().IsVisible()
                                    ? FirstLinked(h, keep, proto.hashed_key)
                                    : h;
      if (first == h) {
        Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
        break;
      }
      if (first == nullptr) {
        unsure = true;
        // Our reference keeps it in its slot; the last release frees it.
        keep->meta.Apply(SlotMeta::VisibleFlag::ClearTransform());
      }
    }
    TEST_SYNC_POINT("FIFOCache::EraseReplaced:Checked");
    replaced = true;
    if (replaced_entries != nullptr) {
      // Our reference keeps h's counters from being reset or counted down,
      // so they only gain hits meanwhile, but a push with no room for h can
      // still move it to the resident queue after we read its word.
      replaced_entries->resident |= !IsInProbation(h);
      replaced_entries->hits =
          std::max(replaced_entries->hits, HitsUnderRef(h));
    }
    if (!TryEraseHandle(h, /*holding_ref=*/true, /*mark_invisible=*/true,
                        /*only_if_invisible=*/false)) {
      TEST_SYNC_POINT("FIFOCache::EraseReplaced:Marked");
      // As in Erase.
      Release(h, /*useful=*/false, /*erase_if_last_ref=*/false);
    }
  }
  return replaced;
}

void AutoHyperClockTable::EraseUnRefEntries() {
  // FIFOCache: up to the claimed slots (see GetClaimedLength).
  size_t usable_size = GetClaimedLength();
  for (size_t i = 0; i < usable_size; i++) {
    HandleImpl& h = array_[i];

    SlotMeta old_meta = h.meta.LoadRelaxed();
    if (old_meta.IsShareable() && old_meta.GetRefcount() == 0) {
      SlotMeta construction_meta;
      construction_meta.SetUnderConstruction();
      if (h.meta.CasStrong(old_meta, construction_meta)) {
        // Took ownership
        h.FreeData(allocator_);
        usage_.FetchSubRelaxed(h.total_charge);
        // FIFOCache: before the slot can be reused.
        TrackRemoval(i, h.total_charge);
        // NOTE: could be more efficient with a dedicated variant of
        // PurgeImpl, but this is not a common operation
        Remove(&h);
        MarkEmpty(h);
        occupancy_.FetchSub(1U);
      }
    }
  }
}

namespace {
// A counter read by Evict less the removals it has pending, at zero or more:
// a step or clock sweep that marks an entry seen pinned adds its charge after
// setting the bit, so an eviction can take the entry out before the add
// lands.
inline size_t SubtractPending(size_t counter, size_t pending) {
  return counter > pending ? counter - pending : 0;
}
}  // namespace

void AutoHyperClockTable::Evict(size_t requested_charge, bool need_entry,
                                InsertState& state, EvictionData* data) {
  // precondition
  assert(requested_charge > 0 || need_entry);
  // FIFOCache: each caller passes a fresh data (but for the insert's
  // probation charge), which this call's evictions, promotions and counter
  // changes are counted from once, on whichever return it takes.
  assert(data->freed_count == 0 && data->probation_freed_count == 0 &&
         data->probation_freed_published == 0 && data->promoted_count == 0 &&
         data->probation_removed_charge == 0 &&
         data->probation_pinned_removed_charge == 0 &&
         data->resident_removed == 0 && !data->effort_cap_reached);
  struct CountOnReturn {
    AutoHyperClockTable& table;
    const EvictionData& data;
    ~CountOnReturn() { table.CountEvictions(data); }
  } count_on_return{*this, *data};

  // We need the clock pointer to seemlessly "wrap around" at the end of the
  // table, and to be reasonably stable under Grow operations. This is
  // challenging when the linear hashing progressively opens additional
  // most-significant-hash-bits in determining home locations.

  // TODO: make a tuning parameter?
  // Up to 2x this number of homes will be evicted per step. In very rare
  // cases, possibly more, as homes of an out-of-date generation will be
  // resolved to multiple in a newer generation.
  constexpr size_t step_size = 4;

  // A clock_pointer_mask_ field separate from length_info_ enables us to use
  // the same mask (way of dividing up the space among evicting threads) for
  // iterating over the whole structure before considering changing the mask
  // at the beginning of each pass. This ensures we do not have a large portion
  // of the space that receives redundant or missed clock updates. However,
  // with two variables, for each update to clock_pointer_mask (< 64 ever in
  // the life of the cache), there will be a brief period where concurrent
  // eviction threads could use the old mask value, possibly causing redundant
  // or missed clock updates for a *small* portion of the table.
  size_t clock_pointer_mask = clock_pointer_mask_.LoadRelaxed();

  uint64_t max_clock_pointer = 0;  // unset

  // TODO: consider updating during a long eviction
  size_t used_length = LengthInfoToUsedLength(state.saved_length_info);

  autovector<HandleImpl*> to_finish_eviction;

  // FIFOCache: S3-FIFO's choice of queue, made before each step. Probation
  // goes first while its entries not last seen pinned hold more than its
  // share, counting the entry being inserted, or when the resident queue
  // holds nothing but cache reservations; otherwise the resident clock does.
  // (Long-lived pins in probation, such as blocks held through the handle
  // their insert returned, would otherwise keep it over its share and leave
  // resident entries unevictable.) Once probation has promoted an entry it
  // keeps going until it evicts one, as S3-FIFO does, even if the promotions
  // took it under its share: otherwise the resident clock would go next and
  // could take the entry just promoted, which has no hits there yet. Once the
  // resident clock has given up (the limit below, or its effort cap),
  // probation goes regardless of its share, and an entry it promotes gets the
  // resident clock one more lap if the clock is under its effort cap, since
  // at no hits it is evictable there.
  // A probation step that finds nothing to pop falls through to a resident
  // step. Probation steps per call are capped at the cells the queue's rings
  // have, a lap of the queue, so heads that keep rotating (losing every claim
  // to hits, which the effort cap doesn't count) cannot keep this call going.
  //
  // The effort cap applies to each queue's steps apart: pinned heads in
  // probation say nothing of the resident queue, nor a resident queue of
  // pinned entries of probation's. A queue whose steps have seen too many
  // pinned entries for the entries this call has freed so far, from either
  // queue (the probation entries the resident clock passes among them),
  // gets no more steps until a free raises its allowance, and the call gives
  // up when both have none.
  //
  // While the clock rests (see clock_laps_), probation goes first, as for a
  // resident queue of reservations alone.
  const size_t probation_target = ProbationTarget();
  bool clock_resting = false;
  if (const uint64_t laps = clock_laps_.LoadRelaxed();
      (laps & kClockRestSet) != 0) {
    const uint64_t evictions = probation_evictions_.LoadRelaxed();
    clock_resting = ClockRestLasts(laps, evictions);
    if (!clock_resting) {
      EndClockRestIfOver(laps, evictions);
    }
  }
  // Read once, the first time probation is over its target.
  size_t pinned_allowance = SIZE_MAX;
  const size_t max_probation_steps = probation_.CellCount();
  size_t probation_steps = 0;
  // The resident clock went round its laps (see max_clock_pointer).
  bool resident_lapped = false;
  bool promoting = false;
  uint64_t resident_laps = uint64_t{ClockHandle::kMaxCountdown + 1};
  size_t probation_seen_pinned = 0;
  size_t resident_seen_pinned = 0;
  auto effort_exceeded = [&](size_t seen_pinned) {
    const uint64_t effort_cap =
        eec_and_scl_.LoadRelaxed().GetEffectiveEvictionEffortCap();
    return (data->freed_count + 1U) * effort_cap <= seen_pinned;
  };
  auto probation_capped = [&] {
    return effort_exceeded(probation_seen_pinned);
  };
  auto resident_capped = [&] { return effort_exceeded(resident_seen_pinned); };

  // Loop until enough freed, or limit reached (see bottom of loop)
  for (;;) {
    // The counters less this call's removals, which CountEvictions takes
    // off on return. (The insert's precharge isn't in them either, and is
    // state.probation_charge here.)
    const size_t probation_usage = SubtractPending(
        probation_usage_.LoadRelaxed(), data->probation_removed_charge);
    const size_t probation_pinned =
        SubtractPending(probation_pinned_usage_.LoadRelaxed(),
                        data->probation_pinned_removed_charge);
    const size_t probation_unpinned = probation_usage > probation_pinned
                                          ? probation_usage - probation_pinned
                                          : 0;
    // Over its share, the target plus the pinned allowance.
    const size_t probation_demand = probation_unpinned + state.probation_charge;
    bool probation_over = probation_demand > probation_target;
    if (probation_over) {
      if (pinned_allowance == SIZE_MAX) {
        pinned_allowance = PinnedAllowance(probation_target);
      }
      probation_over = probation_demand - probation_target > pinned_allowance;
    }
    const bool resident_empty =
        resident_count_.LoadRelaxed() -
            static_cast<int64_t>(data->resident_removed) <=
        0;
    const bool resident_out = resident_lapped || resident_capped();
    const bool probation_first =
        !probation_capped() && (promoting || resident_out || probation_over ||
                                resident_empty || clock_resting);
    bool promoted = false;
    const size_t freed_before = data->freed_count;
    const size_t pinned_before = data->seen_pinned_count;
    if (probation_first && probation_steps < max_probation_steps &&
        (++probation_steps, ProbationStep(data, state, &promoted))) {
      probation_seen_pinned += data->seen_pinned_count - pinned_before;
      if (promoted && resident_lapped) {
        // A lap more, while the clock is under its effort cap.
        resident_lapped = false;
        max_clock_pointer = 0;
        resident_laps = 1;
      }
      // A stale cell, or a head that kept losing races to hits, changes
      // nothing, so the sweep goes on; an eviction or a pinned head (which
      // leaves probation's share) ends it.
      promoting = promoted || (promoting && data->freed_count == freed_before &&
                               data->seen_pinned_count == pinned_before);
      if (data->freed_charge >= requested_charge &&
          (!need_entry || data->freed_count > 0)) {
        return;
      }
      if (probation_capped() &&
          (resident_lapped || resident_capped() || resident_empty)) {
        // FIFOCache: counted by the caller.
        data->effort_cap_reached = true;
        return;
      }
      continue;
    }
    promoting = false;
    if (resident_lapped || resident_capped()) {
      data->effort_cap_reached |= probation_capped() || resident_capped();
      return;
    }
    const size_t resident_pinned_before = data->seen_pinned_count;
    const size_t resident_freed_before = data->freed_count;
    const size_t countdowns_before = data->clock_countdowns;

    TEST_SYNC_POINT("FIFOCache::Evict:ResidentStep");
    // First (concurrent) increment clock pointer
    uint64_t old_clock_pointer = clock_pointer_.FetchAddRelaxed(step_size);
    // FIFOCache: the step's number, and the length in steps of its lap (see
    // clock_laps_), at the mask before any update below.
    const uint32_t step = static_cast<uint32_t>(old_clock_pointer / step_size);
    const uint32_t lap_steps = static_cast<uint32_t>(
        std::max(size_t{1}, (clock_pointer_mask + 1) / step_size));

    if (UNLIKELY((old_clock_pointer & clock_pointer_mask) == 0)) {
      // Back at the beginning. See if clock_pointer_mask should be updated.
      uint64_t mask = BottomNBits(
          UINT64_MAX, LengthInfoToMinShift(state.saved_length_info));
      // FIFOCache: a lap no step of which has evicted or run down a
      // countdown rests the clock, from the rest of this call on, unless
      // the mask changes here or the shard's differs (see clock_laps_).
      const bool judge = clock_pointer_mask == mask &&
                         clock_pointer_mask_.LoadRelaxed() == mask;
      if (clock_pointer_mask != mask) {
        clock_pointer_mask = static_cast<size_t>(mask);
        clock_pointer_mask_.StoreRelaxed(clock_pointer_mask);
      }
      if (judge) {
        uint32_t ended_lap = step - lap_steps;
        TEST_SYNC_POINT_CALLBACK("FIFOCache::Evict:JudgingLap", &ended_lap);
        const uint64_t end = probation_evictions_.LoadRelaxed() +
                             PendingProbationEvictions(*data) +
                             std::min(occupancy_.LoadRelaxed(), kMaxClockRest);
        if (RestClockAfterLap(ended_lap, end)) {
          TEST_SYNC_POINT("FIFOCache::Evict:RestedClock");
          ++data->clock_rests;
          clock_resting = true;
        }
      }
    }

    size_t major_step = clock_pointer_mask + 1;
    assert((major_step & clock_pointer_mask) == 0);

    for (size_t base_home = old_clock_pointer & clock_pointer_mask;
         base_home < used_length; base_home += major_step) {
      for (size_t i = 0; i < step_size; i++) {
        size_t home = base_home + i;
        if (home >= used_length) {
          break;
        }
        PurgeImpl(&to_finish_eviction, home, data);
      }
    }

    for (HandleImpl* h : to_finish_eviction) {
      // FIFOCache: a resident entry, by the check in ClockUpdate.
      TrackEvictionRemoval(static_cast<size_t>(h - array_.Get()),
                           h->total_charge, data);
      TrackAndReleaseEvictedEntry(h);
      // NOTE: setting likely_empty_slot here can cause us to reduce the
      // portion of "at home" entries, probably because an evicted entry
      // is more likely to come back than a random new entry and would be
      // unable to go into its home slot.
    }
    to_finish_eviction.clear();

    // FIFOCache: a step that evicted or ran down a countdown credits its
    // lap, before any return (see clock_laps_).
    if (data->freed_count != resident_freed_before ||
        data->clock_countdowns != countdowns_before) {
      TEST_SYNC_POINT("FIFOCache::Evict:CreditingLap");
      if (CreditClockLap(step & ~(lap_steps - 1))) {
        ++data->clock_rests_called_off;
      }
    }

    // Loop exit conditions
    // FIFOCache: with an entry freed if one is needed.
    if (data->freed_charge >= requested_charge &&
        (!need_entry || data->freed_count > 0)) {
      return;
    }

    if (max_clock_pointer == 0) {
      // Cap the eviction effort at this thread (along with those operating in
      // parallel) circling through the whole structure kMaxCountdown times.
      // In other words, this eviction run must find something/anything that is
      // unreferenced at start of and during the eviction run that isn't
      // reclaimed by a concurrent eviction run.
      // TODO: Does HyperClockCache need kMaxCountdown + 1?
      // FIFOCache: resident_laps, which is that unless probation promoted
      // after the clock gave up.
      max_clock_pointer = old_clock_pointer + resident_laps * major_step;
    }

    if (old_clock_pointer + step_size >= max_clock_pointer) {
      // FIFOCache: fall back on probation, which returns when it has
      // nothing either.
      resident_lapped = true;
    }

    resident_seen_pinned += data->seen_pinned_count - resident_pinned_before;
    if (resident_capped() && probation_capped()) {
      // FIFOCache: counted by the caller.
      data->effort_cap_reached = true;
      return;
    }
  }
}

// FIFOCache: the probation queue and S3-FIFO's steps.

ProbationQueue::ProbationQueue(size_t first_ring_capacity,
                               size_t max_ring_capacity)
    // Ring capacities are distinct powers of two up to max_ring_capacity, so
    // all of them fit in twice that.
    : cells_(sizeof(Cell) * 2 * max_ring_capacity),
      max_ring_capacity_(max_ring_capacity),
      max_rings_(static_cast<uint32_t>(FloorLog2(max_ring_capacity) -
                                       FloorLog2(first_ring_capacity) + 1)),
      rings_(new Ring[max_rings_]) {
  assert(first_ring_capacity > 0);
  assert((first_ring_capacity & (first_ring_capacity - 1)) == 0);
  assert((max_ring_capacity & (max_ring_capacity - 1)) == 0);
  assert(first_ring_capacity <= max_ring_capacity);
  if (cells_.Get() == nullptr) {
    fprintf(stderr, "Anonymous mmap for RocksDB FIFOCache failed. Aborting.\n");
    std::terminate();
  }
  rings_[0].cells = cells_.Get();
  rings_[0].mask = first_ring_capacity - 1;
  rings_[0].shift = FloorLog2(first_ring_capacity);
}

bool ProbationQueue::Push(uint64_t v) {
  assert(v <= kMaxValue);
  for (;;) {
    const uint32_t t = tail_.load(std::memory_order_acquire);
    Ring& r = rings_[t];
    // Acquire, so that a closed flag seen here comes with the newer tail_,
    // and the cell below is no older than the claim of its position a lap
    // before found it (see Cell).
    uint64_t pos = r.enqueue.load(std::memory_order_acquire);
    TEST_SYNC_POINT("FIFOCache::ProbationQueue::Push:Loaded");
    while ((pos & kClosed) == 0) {
      Cell& c = r.cells[pos & r.mask];
      const uint64_t lap = pos >> r.shift;
      if ((c.load(std::memory_order_acquire) & kPhaseMask) ==
          Phase(lap, /*filled=*/false)) {
        if (r.enqueue.compare_exchange_weak(pos, pos + 1,
                                            std::memory_order_acq_rel)) {
          TEST_SYNC_POINT("FIFOCache::ProbationQueue::Push:Claimed");
          c.store((v << kPhaseBits) | Phase(lap, /*filled=*/true),
                  std::memory_order_release);
          return true;
        }
        // pos now holds the current enqueue position.
      } else {
        // Full, or the pop of this cell's last value hasn't finished, if pos
        // hasn't moved. Otherwise another push took pos, or the ring was
        // closed (and perhaps drained and zeroed) since pos was loaded.
        const uint64_t cur = r.enqueue.load(std::memory_order_acquire);
        if (cur == pos) {
          return false;
        }
        pos = cur;
      }
    }
    // Closed: a newer ring takes pushes. tail_ moved before the close.
    if (tail_.load(std::memory_order_acquire) == t) {
      assert(false);
      return false;
    }
  }
}

bool ProbationQueue::PopFromRing(Ring& r, uint64_t* v) {
  // Acquire, so that the cell below is no older than the claim of its
  // position a lap before found it (see Cell).
  uint64_t pos = r.dequeue.load(std::memory_order_acquire);
  TEST_SYNC_POINT("FIFOCache::ProbationQueue::PopFromRing:Loaded");
  for (;;) {
    Cell& c = r.cells[pos & r.mask];
    const uint64_t lap = pos >> r.shift;
    const uint64_t cell = c.load(std::memory_order_acquire);
    if ((cell & kPhaseMask) == Phase(lap, /*filled=*/true)) {
      // Filled at pos, and stays so until popped, so the value read here is
      // the one a successful claim below takes.
      if (r.dequeue.compare_exchange_weak(pos, pos + 1,
                                          std::memory_order_acq_rel)) {
        TEST_SYNC_POINT("FIFOCache::ProbationQueue::PopFromRing:Claimed");
        *v = cell >> kPhaseBits;
        c.store(Phase(lap + 1, /*filled=*/false), std::memory_order_release);
        return true;
      }
      // pos now holds the current dequeue position.
    } else {
      // Nothing pushed at pos yet, or the push hasn't filled the cell, if
      // pos hasn't moved. Otherwise another pop took pos.
      const uint64_t cur = r.dequeue.load(std::memory_order_acquire);
      if (cur == pos) {
        return false;
      }
      pos = cur;
    }
  }
}

bool ProbationQueue::Pop(uint64_t* v, size_t* drained_cells) {
  for (;;) {
    uint32_t h = head_.load(std::memory_order_acquire);
    TEST_SYNC_POINT("FIFOCache::ProbationQueue::Pop:Loaded");
    Ring& r = rings_[h];
    if (PopFromRing(r, v)) {
      return true;
    }
    if (tail_.load(std::memory_order_acquire) == h) {
      return false;
    }
    // A newer ring exists. Close this one if the thread adding the newer
    // ring hasn't yet, so that no push lands here after the check below.
    uint64_t e = r.enqueue.load(std::memory_order_acquire);
    if ((e & kClosed) == 0) {
      e = r.enqueue.fetch_or(kClosed, std::memory_order_acq_rel) | kClosed;
    }
    if (r.dequeue.load(std::memory_order_acquire) != (e & ~kClosed)) {
      // A push to this ring hasn't filled its cell yet.
      return false;
    }
    // Every position pushed here has been taken: move on (or see that
    // another thread did). Only the thread that moves on drops the ring.
    if (head_.compare_exchange_strong(h, h + 1, std::memory_order_acq_rel)) {
      DropRing(r);
      *drained_cells += static_cast<size_t>(r.mask + 1);
    }
  }
}

void ProbationQueue::DropRing(const Ring& r) {
  Cell* const begin = r.cells;
  Cell* const end = r.cells + r.mask + 1;
  // The whole pages inside the ring, which no other ring shares.
  Cell* dropped_begin = end;
  Cell* dropped_end = end;
#ifdef OS_LINUX
  // Linux fills a dropped page of a private anonymous mapping with zeros on
  // its next touch, and a read maps the shared zero page.
  static const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t b =
      (reinterpret_cast<uintptr_t>(begin) + page - 1) & ~(page - 1);
  const uintptr_t e = reinterpret_cast<uintptr_t>(end) & ~(page - 1);
  if (b < e && madvise(reinterpret_cast<void*>(b), e - b, MADV_DONTNEED) == 0) {
    dropped_begin = reinterpret_cast<Cell*>(b);
    dropped_end = reinterpret_cast<Cell*>(e);
  }
#endif
  for (Cell* c = begin; c < end; ++c) {
    if (c == dropped_begin) {
      c = dropped_end - 1;
      continue;
    }
    c->store(0, std::memory_order_relaxed);
  }
}

bool ProbationQueue::PopNewest(uint64_t* v) {
  // Acquire, pairing with TryAddRing's release, so the ring's cells and mask
  // are visible.
  return PopFromRing(rings_[tail_.load(std::memory_order_acquire)], v);
}

size_t ProbationQueue::RingCapacity() const {
  return static_cast<size_t>(
      rings_[tail_.load(std::memory_order_acquire)].mask + 1);
}

bool ProbationQueue::TryAddRing(size_t capacity) {
  const uint32_t t = tail_.load(std::memory_order_acquire);
  const Ring& last = rings_[t];
  assert((capacity & (capacity - 1)) == 0);
  if (capacity <= last.mask + 1 || capacity > max_ring_capacity_ ||
      t + 1 >= max_rings_) {
    return false;
  }
  uint32_t expected = t + 1;
  if (!rings_claimed_.compare_exchange_strong(expected, t + 2,
                                              std::memory_order_acq_rel)) {
    return false;
  }
  Ring& next = rings_[t + 1];
  next.cells = last.cells + last.mask + 1;
  next.mask = capacity - 1;
  next.shift = FloorLog2(capacity);
  assert(next.cells + capacity <= cells_.Get() + cells_.Count());
  tail_.store(t + 1, std::memory_order_release);
  TEST_SYNC_POINT("FIFOCache::ProbationQueue::TryAddRing:BeforeClose");
  rings_[t].enqueue.fetch_or(kClosed, std::memory_order_acq_rel);
  return true;
}

size_t ProbationQueue::CellCount() const {
  const Ring& last = rings_[tail_.load(std::memory_order_acquire)];
  return static_cast<size_t>(last.cells - cells_.Get()) + last.mask + 1;
}

size_t ProbationQueue::LiveCellCount() const {
  const Ring& first = rings_[head_.load(std::memory_order_acquire)];
  return CellCount() - static_cast<size_t>(first.cells - cells_.Get());
}

size_t AutoHyperClockTable::ProbationRingCapacity(size_t occupancy_limit) {
  // The smallest power of two holding every entry the table can.
  return occupancy_limit <= 1 ? size_t{1}
                              : size_t{2} << FloorLog2(occupancy_limit - 1);
}

size_t AutoHyperClockTable::ProbationTarget() const {
  // Exact without overflowing on large capacities.
  const size_t capacity = capacity_.LoadRelaxed();
  return capacity / 100 * kProbationPercent +
         capacity % 100 * kProbationPercent / 100;
}

size_t AutoHyperClockTable::PinnedAllowance(size_t probation_target) const {
  // Capped again here, for a capacity cut since it last changed.
  return std::min(pinned_allowance_.LoadRelaxed(),
                  kPinnedAllowanceTargets * probation_target);
}

bool AutoHyperClockTable::TryInsertSlot(const ClockHandleBasicData& proto,
                                        size_t idx, uint32_t initial_countdown,
                                        uint32_t word_flags,
                                        bool probation_precharged,
                                        uint32_t* word) {
  HandleImpl& h = array_[idx];
  if (!BeginSlotInsert(h)) {
    return false;
  }
  // We own the slot, so its word is ours to write.
  RelaxedAtomic<uint32_t>& w = probation_words_[idx];
  const uint32_t old_word = w.LoadRelaxed();
  assert((old_word & kInProbation) == 0);
  *word =
      ((old_word & ~(kProbationGenerationOne - 1)) + kProbationGenerationOne) |
      word_flags;
  w.StoreRelaxed(*word);
  if (word_flags & kInProbation) {
    if (!probation_precharged) {
      probation_usage_.FetchAddRelaxed(proto.GetTotalCharge());
    }
  } else {
    assert(!probation_precharged);
    if (!(word_flags & kReservation)) {
      resident_count_.FetchAddRelaxed(1);
    }
  }
  // Publishes the word with the entry (a release store).
  FinishSlotInsert(proto, h, initial_countdown, /*keep_ref=*/true);
  return true;
}

bool AutoHyperClockTable::ProbationPush(size_t idx, uint32_t word,
                                        InsertState& state, bool rotated) {
  const uint64_t v = ProbationCell(idx, word) | (rotated ? kRotatedCell : 0);
  if (probation_.Push(v)) {
    return true;
  }
  const size_t want = ProbationRingCapacity(occupancy_limit_.LoadRelaxed());
  if (want > probation_.RingCapacity()) {
    // Charged first, so the strict limit can turn the ring down; then this
    // entry goes to the resident queue below unless a ring has room.
    const size_t ring_bytes = want * sizeof(ProbationQueue::Cell);
    if (ChargeMetadata(ring_bytes, /*keep=*/0)) {
      TEST_SYNC_POINT("FIFOCache::ProbationPush:RingCharged");
      if (probation_.TryAddRing(want)) {
        if (metadata_charge_policy_ == kFullChargeCacheMetadata) {
          state.ring_charge += ring_bytes;
        }
      } else {
        UnchargeMetadata(ring_bytes);
      }
    }
    // Whether this call or another added it, or one is still being added.
    if (probation_.Push(v)) {
      return true;
    }
    // The table has outgrown the newest ring, which can then hold only live
    // cells, so this entry goes to the resident queue.
    return false;
  }
  // Full, though the table hasn't outgrown the newest ring. The cells of
  // entries removed while queued stay until popped, and pops reach the
  // newest ring only once the older rings are empty, which they never are
  // while the cache fills and nothing is evicted. So make room in the
  // newest ring itself: take up to kMaxRingPopsPerPush cells from its
  // oldest end, put live ones back at its newest end, and push this entry
  // in place of the first stale one. A live entry put back moves behind the
  // live entries that were behind it in that ring, or behind all of a ring
  // added meanwhile. If no stale cell turns up, this entry goes to the
  // resident queue.
  //
  // A live cell whose room another push took first is carried, and goes in
  // place of the next stale cell, ahead of this entry, rather than leaving
  // probation for the newer entry's sake. A second one, or a carried one
  // that finds no stale cell, goes to the resident queue.
  auto leave = [this](uint64_t cell) {
    if (LeaveProbationUnowned(ProbationCellSlot(cell),
                              ProbationCellWord(cell))) {
      probation_full_.FetchAddRelaxed(1);
    }
  };
  bool carrying = false;
  uint64_t carried = 0;
  for (size_t pops = 0; pops < kMaxRingPopsPerPush; ++pops) {
    uint64_t cell;
    if (!probation_.PopNewest(&cell)) {
      break;
    }
    if (!IsStaleProbationCell(cell)) {
      TEST_SYNC_POINT("FIFOCache::ProbationPush:BeforeRequeue");
      // The pop freed the cell this push takes, unless another push took
      // it first.
      if (!probation_.Push(cell)) {
        if (carrying) {
          leave(cell);
        } else {
          carrying = true;
          carried = cell;
        }
      }
      continue;
    }
    TEST_SYNC_POINT("FIFOCache::ProbationPush:BeforeReplace");
    // Another push can take the room first; then look on.
    if (carrying) {
      carrying = !probation_.Push(carried);
    } else if (probation_.Push(v)) {
      return true;
    }
  }
  if (carrying) {
    leave(carried);
  }
  return false;
}

size_t AutoHyperClockTable::GhostGroup(const UniqueId64x2& hashed_key) const {
  size_t home;
  int shift;
  GetHomeIndexAndShift(length_info_.LoadRelaxed(), hashed_key[1], &home,
                       &shift);
  // Inside the mappings, whose slot count is a multiple of 64.
  return home & ~(kGhostWays - 1);
}

bool AutoHyperClockTable::ClockResting() const {
  return ClockRestLasts(clock_laps_.LoadRelaxed(),
                        probation_evictions_.LoadRelaxed());
}

bool AutoHyperClockTable::ClockRestLasts(uint64_t laps, uint64_t evictions) {
  if ((laps & kClockRestSet) == 0) {
    return false;
  }
  const uint32_t left = (static_cast<uint32_t>(laps >> kClockRestEndShift) -
                         static_cast<uint32_t>(evictions)) &
                        kClockRestEndMask;
  return left != 0 && left <= (kClockRestEndMask >> 1);
}

void AutoHyperClockTable::EndClockRestIfOver(uint64_t laps,
                                             uint64_t evictions) {
  if ((laps & kClockRestSet) != 0 && !ClockRestLasts(laps, evictions)) {
    TEST_SYNC_POINT("FIFOCache::EndClockRestIfOver:Over");
    // Fails if the word has changed, which only a credit, a rest or a clear
    // does, so it holds this rest no longer.
    clock_laps_.CasStrongRelaxed(laps, laps & kClockLapMask);
  }
}

bool AutoHyperClockTable::RestClockAfterLap(uint32_t lap, uint64_t end) {
  uint64_t laps = clock_laps_.LoadRelaxed();
  for (;;) {
    const uint32_t credited = static_cast<uint32_t>(laps & kClockLapMask);
    if (static_cast<int32_t>(credited - lap) >= 0) {
      return false;
    }
    // Credited up to the step before the lap, which it was not.
    const uint64_t rested =
        uint64_t{lap - 1} | kClockRestSet |
        (uint64_t{static_cast<uint32_t>(end) & kClockRestEndMask}
         << kClockRestEndShift);
    if (clock_laps_.CasWeakRelaxed(laps, rested)) {
      return true;
    }
  }
}

bool AutoHyperClockTable::CreditClockLap(uint32_t lap) {
  uint64_t laps = clock_laps_.LoadRelaxed();
  for (;;) {
    const uint32_t credited = static_cast<uint32_t>(laps & kClockLapMask);
    if (static_cast<int32_t>(lap - credited) <= 0) {
      return false;
    }
    const bool called_off =
        lap == credited + 1 &&
        ClockRestLasts(laps, probation_evictions_.LoadRelaxed());
    if (clock_laps_.CasWeakRelaxed(laps, uint64_t{lap})) {
      return called_off;
    }
  }
}

uint32_t AutoHyperClockTable::GhostTag(const UniqueId64x2& hashed_key) {
  // The home comes from hashed_key[1], so the tag from the other half.
  const uint32_t tag = static_cast<uint32_t>(hashed_key[0] & 0xffff);
  return tag == 0 ? 1 : tag;
}

uint32_t AutoHyperClockTable::GhostNow(int shift_word, size_t pending) const {
  return static_cast<uint32_t>((probation_evictions_.LoadRelaxed() + pending) >>
                               (shift_word & kGhostShiftMask)) &
         0xffff;
}

uint32_t AutoHyperClockTable::GhostReach(int shift_word) const {
  const int shift = shift_word & kGhostShiftMask;
  const size_t step = size_t{1} << shift;
  // As many keys as the resident queue's share of the entries: all but
  // probation's, rounded down as ProbationTarget rounds it.
  const size_t entries = occupancy_.LoadRelaxed();
  const size_t resident_share =
      entries - (entries / 100 * kProbationPercent +
                 entries % 100 * kProbationPercent / 100);
  const size_t steps = (resident_share + step - 1) >> shift;
  // Racing inserts can take occupancy a little past its limit.
  return static_cast<uint32_t>(std::min(steps, size_t{1} << 15));
}

uint32_t AutoHyperClockTable::GhostAge(uint32_t cell, uint32_t now) {
  const uint32_t age = (now - cell) & 0xffff;
  // A stamp a little ahead of now was written by a pass that read the
  // count after the caller did.
  return age > 0xffff - kGhostStampSlack ? 0 : age;
}

uint32_t AutoHyperClockTable::GhostRestamp(uint32_t c, uint32_t old_now,
                                           int old_shift, uint32_t now,
                                           int shift) {
  const uint32_t ahead = (c - old_now) & 0xffff;
  if (ahead != 0 && ahead <= kGhostStampSlack) {
    // Stamped ahead of the count by a call whose evictions aren't all in it
    // yet (see GhostNow), so it keeps its lead, in the new step and within
    // the slack.
    const uint64_t lead = shift > old_shift
                              ? ahead >> (shift - old_shift)
                              : uint64_t{ahead} << (old_shift - shift);
    return (c & 0xffff0000) | ((now + static_cast<uint32_t>(std::min(
                                          lead, uint64_t{kGhostStampSlack}))) &
                               0xffff);
  }
  const uint64_t old_age = GhostAge(c, old_now);
  const uint64_t age = shift > old_shift ? old_age >> (shift - old_shift)
                                         : old_age << (old_shift - shift);
  return age >= (uint64_t{1} << 15)
             ? 0
             : (c & 0xffff0000) | ((now - static_cast<uint32_t>(age)) & 0xffff);
}

size_t AutoHyperClockTable::GhostFind(size_t group, uint32_t tag, size_t from,
                                      uint32_t* cell, int* shift_word) const {
  uint32_t now = 0;
  uint32_t reach = 0;
  bool loaded = false;
  for (size_t i = from; i < kGhostWays; ++i) {
    const uint32_t c = ghost_[group + i].LoadRelaxed();
    if ((c >> 16) != tag) {
      continue;
    }
    // Only for a matching tag, which most lookups don't find.
    if (!loaded) {
      if (*shift_word < 0) {
        *shift_word = ghost_stamp_shift_.LoadRelaxed();
      }
      if (*shift_word & kGhostRestamping) {
        return kGhostWays;
      }
      now = GhostNow(*shift_word);
      reach = GhostReach(*shift_word);
      loaded = true;
    }
    if (GhostAge(c, now) < reach) {
      *cell = c;
      return i;
    }
  }
  return kGhostWays;
}

bool AutoHyperClockTable::GhostPlace(size_t group, uint32_t value,
                                     uint32_t now) {
  // Over a cell older than value, the oldest, if there is no empty one.
  size_t oldest = kGhostWays;
  uint32_t oldest_cell = 0;
  uint32_t oldest_age = GhostAge(value, now);
  for (size_t i = 0; i < kGhostWays; ++i) {
    uint32_t c = ghost_[group + i].LoadRelaxed();
    if (c == 0) {
      if (ghost_[group + i].CasStrongRelaxed(c, value)) {
        return true;
      }
      // Taken meanwhile; c is what took it.
    }
    const uint32_t age = GhostAge(c, now);
    if (age > oldest_age) {
      oldest = i;
      oldest_cell = c;
      oldest_age = age;
    }
  }
  // A racing write to that cell wins.
  return oldest < kGhostWays &&
         ghost_[group + oldest].CasStrongRelaxed(oldest_cell, value);
}

void AutoHyperClockTable::GhostTake(const ClockHandleBasicData& proto,
                                    InsertState& state) {
  state.ghost_hit = false;
  if (IsPlaceholder(proto)) {
    return;
  }
  const uint32_t tag = GhostTag(proto.hashed_key);
  const size_t group = GhostGroup(proto.hashed_key);
  uint32_t cell;
  int shift_word = -1;
  for (size_t i = 0;
       (i = GhostFind(group, tag, i, &cell, &shift_word)) < kGhostWays; ++i) {
    if (ghost_[group + i].CasStrongRelaxed(cell, 0)) {
      state.ghost_hit = true;
      state.ghost_value = cell;
      state.ghost_shift = shift_word;
      return;
    }
  }
}

void AutoHyperClockTable::GhostPutBack(const ClockHandleBasicData& proto,
                                       const InsertState& state) {
  if (!state.ghost_hit) {
    return;
  }
  const int shift_word = ghost_stamp_shift_.LoadRelaxed();
  if (shift_word & kGhostRestamping) {
    // In either step to the restamp, so dropped.
    return;
  }
  const uint32_t now = GhostNow(shift_word);
  uint32_t value = state.ghost_value;
  if (shift_word != state.ghost_shift) {
    // Restamped meanwhile, so converted as the restamp would have.
    value = GhostRestamp(value, GhostNow(state.ghost_shift), state.ghost_shift,
                         now, shift_word);
    if (value == 0) {
      return;
    }
  }
  // In the key's group now, which growth may have changed, with the stamp
  // it had.
  GhostPlace(GhostGroup(proto.hashed_key), value, now);
}

void AutoHyperClockTable::GhostPut(const UniqueId64x2& hashed_key,
                                   size_t pending) {
  const int shift_word = ghost_stamp_shift_.LoadRelaxed();
  if (shift_word & kGhostRestamping) {
    return;
  }
  const uint32_t now = GhostNow(shift_word, pending);
  TEST_SYNC_POINT("FIFOCache::GhostPut:Loaded");
  GhostPlace(GhostGroup(hashed_key), (GhostTag(hashed_key) << 16) | now, now);
}

void AutoHyperClockTable::GhostMerge(size_t from, size_t to) {
  const int shift_word = ghost_stamp_shift_.LoadRelaxed();
  if (shift_word & kGhostRestamping) {
    return;
  }
  const uint32_t now = GhostNow(shift_word);
  const uint32_t reach = GhostReach(shift_word);
  for (size_t i = 0; i < kGhostWays; ++i) {
    const uint32_t c = ghost_[from + i].LoadRelaxed();
    const uint32_t age = GhostAge(c, now);
    if (c == 0 || age >= reach) {
      continue;
    }
    bool there = false;
    for (size_t j = 0; j < kGhostWays && !there; ++j) {
      const uint32_t d = ghost_[to + j].LoadRelaxed();
      there = (d >> 16) == (c >> 16) && GhostAge(d, now) <= age;
    }
    if (!there) {
      GhostPlace(to, c, now);
    }
  }
}

void AutoHyperClockTable::SetCapacity(size_t capacity) {
  BaseClockTable::SetCapacity(capacity);
  const int shift =
      GhostStampShift(CalcOccupancyLimit(CalcMaxUsableLength(capacity)));
  // SetCapacity calls are serialized (ShardedCache's config_mutex_), so
  // the word has no kGhostRestamping here.
  const int old_shift = ghost_stamp_shift_.LoadRelaxed();
  if (shift == old_shift) {
    return;
  }
  ghost_stamp_shift_.StoreRelaxed(shift | kGhostRestamping);
  TEST_SYNC_POINT("FIFOCache::SetCapacity:Restamping");
  const uint64_t evictions = probation_evictions_.LoadRelaxed();
  const uint32_t old_now =
      static_cast<uint32_t>(evictions >> old_shift) & 0xffff;
  const uint32_t now = static_cast<uint32_t>(evictions >> shift) & 0xffff;
  // Restamps every cell of every group a home can be in, up to the slots
  // growth has claimed (a Grow fills the new slot's group before the used
  // length counts it), with its age in the new step (rounded down),
  // dropping the tags too old to count in it, so a capacity that moves back
  // and forth across a step keeps the ghost. A tag stamped ahead of the
  // count keeps its lead (see GhostRestamp).
  const size_t end = std::min(grow_frontier_.LoadRelaxed(), array_.Count());
  for (size_t group = 0; group < end; group += kGhostWays) {
    for (size_t i = 0; i < kGhostWays; ++i) {
      RelaxedAtomic<uint32_t>& cell = ghost_[group + i];
      uint32_t c = cell.LoadRelaxed();
      if (c == 0) {
        continue;
      }
      // A write that read the old step before the flag and lands over c
      // stays as written.
      cell.CasStrongRelaxed(c, GhostRestamp(c, old_now, old_shift, now, shift));
    }
  }
  ghost_stamp_shift_.StoreRelaxed(shift);
}

bool AutoHyperClockTable::InGhost(const UniqueId64x2& hashed_key) const {
  uint32_t cell;
  int shift_word = -1;
  return GhostFind(GhostGroup(hashed_key), GhostTag(hashed_key), 0, &cell,
                   &shift_word) < kGhostWays;
}

bool AutoHyperClockTable::IsInProbation(const HandleImpl* h) const {
  if (h->IsStandalone()) {
    return false;
  }
  return (probation_words_[static_cast<size_t>(h - array_.Get())]
              .LoadRelaxed() &
          kInProbation) != 0;
}

bool AutoHyperClockTable::CountsInsertRef(const HandleImpl* h) const {
  if (h->IsStandalone()) {
    return false;
  }
  return (probation_words_[static_cast<size_t>(h - array_.Get())]
              .LoadRelaxed() &
          kCountsInsertRef) != 0;
}

uint32_t AutoHyperClockTable::HitsUnderRef(const HandleImpl* h) const {
  assert(!h->IsStandalone());
  // If the clock took the insert's count out just before the caller's
  // reference, the bit may still read as set, and this comes up one short.
  const uint32_t w =
      probation_words_[static_cast<size_t>(h - array_.Get())].LoadRelaxed();
  const uint32_t not_hits = (w & kCountsInsertRef) ? 2 : 1;
  const uint32_t acquires = h->meta.Load().GetAcquireCounter();
  return acquires > not_hits ? acquires - not_hits : 0;
}

bool AutoHyperClockTable::IsStaleProbationCell(uint64_t v) const {
  // The word's identity only moves on (kInProbation cleared, or a later
  // entry's generation), so a cell found stale stays stale.
  return (probation_words_[ProbationCellSlot(v)].LoadRelaxed() &
          kProbationIdentity) != ProbationCellWord(v);
}

bool AutoHyperClockTable::LeaveProbation(size_t idx, uint32_t word,
                                         size_t total_charge,
                                         bool counters_reset) {
  assert(word & kInProbation);
  RelaxedAtomic<uint32_t>& w = probation_words_[idx];
  const uint32_t clear =
      kInProbation | kSeenPinned | (counters_reset ? kCountsInsertRef : 0);
  // Counted resident first, so that a removal seeing kInProbation clear
  // never takes the count below what it was.
  resident_count_.FetchAddRelaxed(1);
  uint32_t cur = w.LoadRelaxed();
  while ((cur & kProbationIdentity) == (word & kProbationIdentity)) {
    if (w.CasWeakRelaxed(cur, cur & ~clear)) {
      probation_usage_.FetchSubRelaxed(total_charge);
      if (cur & kSeenPinned) {
        probation_pinned_usage_.FetchSubRelaxed(total_charge);
      }
      return true;
    }
  }
  resident_count_.FetchSubRelaxed(1);
  return false;
}

uint32_t AutoHyperClockTable::TakeReplacedPlace(size_t idx, uint32_t word,
                                                const ReplacedEntries& replaced,
                                                size_t total_charge) {
  // Erased meanwhile: whoever drops the last reference frees it. It stays in
  // probation, so that if a scan drops that reference, the next step to
  // reach it frees it, rather than the resident clock. (An erase right
  // after this check can leave it to the clock, whose first visit frees
  // it.)
  if (!array_[idx].meta.Load().IsVisible()) {
    return word;
  }
  // The insert counts as one more hit: a key inserted while cached has been
  // accessed again (two readers that missed one block).
  const uint32_t hits =
      std::min(replaced.hits + 1, uint32_t{ClockHandle::kMaxCountdown});
  array_[idx].meta.Apply(AcquireCounter::PlusTransformPromiseNoOverflow(hits) +
                         ReleaseCounter::PlusTransformPromiseNoOverflow(hits));
  // Not queued yet, so nothing else moves it out of probation; an owner
  // removing it (once our reference is gone) clears kInProbation itself.
  if (replaced.resident && (word & kInProbation) &&
      LeaveProbation(idx, word, total_charge)) {
    return word & ~(kInProbation | kSeenPinned);
  }
  return word;
}

void AutoHyperClockTable::TrackRemoval(size_t idx, size_t total_charge) {
  // The word, not the entry's helper, which an erase has freed with a key
  // kept on the heap.
  const uint32_t old_word =
      probation_words_[idx].FetchAndRelaxed(~(kInProbation | kSeenPinned));
  if (old_word & kInProbation) {
    probation_usage_.FetchSubRelaxed(total_charge);
    if (old_word & kSeenPinned) {
      probation_pinned_usage_.FetchSubRelaxed(total_charge);
    }
  } else if (!(old_word & kReservation)) {
    resident_count_.FetchSubRelaxed(1);
  }
}

bool AutoHyperClockTable::TrackEvictionRemoval(size_t idx, size_t total_charge,
                                               EvictionData* data) {
  // The word is cleared now, so nothing moves the entry between the
  // counters' parts once it is gone; the counters change on return.
  const uint32_t old_word =
      probation_words_[idx].FetchAndRelaxed(~(kInProbation | kSeenPinned));
  if (old_word & kInProbation) {
    data->probation_removed_charge += total_charge;
    if (old_word & kSeenPinned) {
      data->probation_pinned_removed_charge += total_charge;
    }
    return true;
  }
  if (!(old_word & kReservation)) {
    data->resident_removed += 1;
  }
  return false;
}

void AutoHyperClockTable::UndoProbationPrecharge(const InsertState& state) {
  if (state.probation_precharged) {
    probation_usage_.FetchSubRelaxed(state.probation_charge);
  }
}

template <typename Fn>
void AutoHyperClockTable::WithReference(size_t idx, const Fn& fn) {
  HandleImpl& h = array_[idx];
  SlotMeta old_meta;
  h.meta.Apply(AcquireCounter::PlusTransformPromiseNoOverflow(1), &old_meta);
  if (!old_meta.IsShareable()) {
    // Empty or under (de)construction, where the acquire counter is
    // overwritten before use, as in Lookup; an owner removing the entry
    // clears kInProbation itself.
    return;
  }
  // Also when invisible (erased while referenced).
  TEST_SYNC_POINT("FIFOCache::WithReference:Held");
  fn(h);
  // Frees it if it was erased meanwhile and this was the last reference.
  Release(&h, /*useful=*/false, /*erase_if_last_ref=*/false);
}

size_t AutoHyperClockTable::MarkSeenPinned(size_t idx, uint32_t word) {
  size_t marked = 0;
  WithReference(idx, [&](HandleImpl& h) {
    RelaxedAtomic<uint32_t>& w = probation_words_[idx];
    uint32_t cur = w.LoadRelaxed();
    while ((cur & kProbationIdentity) == (word & kProbationIdentity) &&
           (cur & kSeenPinned) == 0) {
      if (w.CasWeakRelaxed(cur, cur | kSeenPinned)) {
        probation_pinned_usage_.FetchAddRelaxed(h.total_charge);
        marked = h.total_charge;
        break;
      }
    }
  });
  return marked;
}

void AutoHyperClockTable::SweepProbationEntry(HandleImpl& h, uint32_t word,
                                              bool pinned) {
  if (pinned == ((word & kSeenPinned) != 0)) {
    return;
  }
  // Under the chain's lock the entry cannot leave its slot (removal unlinks
  // it under this lock first), so its charge is stable.
  RelaxedAtomic<uint32_t>& w =
      probation_words_[static_cast<size_t>(&h - array_.Get())];
  uint32_t expected = word;
  if (w.CasStrongRelaxed(expected, word ^ kSeenPinned)) {
    if (pinned) {
      probation_pinned_usage_.FetchAddRelaxed(h.total_charge);
    } else {
      probation_pinned_usage_.FetchSubRelaxed(h.total_charge);
    }
  }
}

void AutoHyperClockTable::DropInsertRefCount(HandleImpl& h,
                                             RelaxedAtomic<uint32_t>& word,
                                             uint32_t w) {
  SlotMeta meta = h.meta.Load();
  const uint32_t acquire_count = meta.GetAcquireCounter();
  if (acquire_count != meta.GetReleaseCounter()) {
    // Pinned, perhaps by that reference still.
    return;
  }
  // An entry being removed has no counts, so only the bit is cleared.
  TEST_SYNC_POINT("FIFOCache::DropInsertRefCount:Loaded");
  if (acquire_count > 0) {
    SlotMeta less = meta;
    less.SetAcquireCounter(acquire_count - 1);
    less.SetReleaseCounter(acquire_count - 1);
    if (!h.meta.CasStrong(meta, less)) {
      // Used meanwhile; a later visit tries again.
      return;
    }
  }
  // Under the chain's lock the entry cannot leave its slot, so only an
  // owner removing it can have changed the word.
  word.CasStrongRelaxed(w, w & ~kCountsInsertRef);
}

bool AutoHyperClockTable::ProbationStep(EvictionData* data, InsertState& state,
                                        bool* promoted) {
  uint64_t v;
  size_t drained_cells = 0;
  const bool popped = probation_.Pop(&v, &drained_cells);
  if (drained_cells > 0) {
    // Exactly once per ring, by the pop that moved past it.
    UnchargeMetadata(drained_cells * sizeof(ProbationQueue::Cell));
  }
  if (!popped) {
    return false;
  }
  const size_t idx = ProbationCellSlot(v);
  const uint32_t word = ProbationCellWord(v);
  HandleImpl& h = array_[idx];
  RelaxedAtomic<uint32_t>& w = probation_words_[idx];
  // Lookups and Releases change the counters under us, so a lost race is
  // retried a few times before the head gets another lap. The entry's
  // charge is read only while holding a reference or ownership, since
  // without one the slot can be emptied and reused at any time.
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Acquire, pairing with the publishing store in TryInsertSlot, so a
    // later entry in the slot shows its own word.
    SlotMeta meta = h.meta.Load();
    const uint32_t cur = w.LoadRelaxed();
    TEST_SYNC_POINT("FIFOCache::ProbationStep:Loaded");
    if ((cur & kProbationIdentity) != (word & kProbationIdentity) ||
        !meta.IsShareable()) {
      // Stale: the entry left probation, or is being removed by an owner
      // that will clear kInProbation itself.
      return true;
    }
    const uint32_t acquire_count = meta.GetAcquireCounter();
    if (acquire_count != meta.GetReleaseCounter()) {
      // Seen pinned either way, against the effort cap, so that the work of
      // one call stays bounded when probation's heads are all pinned.
      data->seen_pinned_count++;
      if (v & kRotatedCell) {
        // Pinned again a lap after a step rotated it for being pinned: held
        // for longer than the queue's laps, as a table reader holds pinned
        // index and filter blocks, or hit often enough that lookups hold it
        // whenever a step comes by. Each further lap would cost a step
        // counted against the effort cap, so it goes to the resident queue,
        // whose clock passes over pinned entries as HyperClockCache's does.
        if (LeaveProbationUnowned(idx, word)) {
          data->pinned_twice_count += 1;
        }
        return true;
      }
      // Pinned: rotate it to the tail, out of probation's share until a
      // step or sweep sees it unpinned.
      if ((cur & kSeenPinned) == 0) {
        const size_t marked = MarkSeenPinned(idx, word);
        if ((cur & kCountsInsertRef) && acquire_count == 1) {
          // Held by its inserter alone, so no lookup has hit it: a longer
          // queue would have outlasted the pin (see pinned_allowance_).
          data->allowance_grow_charge += marked;
        }
      }
      if (!ProbationPush(idx, word, state, /*rotated=*/true) &&
          LeaveProbationUnowned(idx, word)) {
        probation_full_.FetchAddRelaxed(1);
      }
      return true;
    }
    const bool counts_insert_ref =
        (cur & kCountsInsertRef) && acquire_count > 0;
    const uint32_t hits = acquire_count - (counts_insert_ref ? 1 : 0);
    if (meta.IsVisible() && hits >= kPromoteThreshold) {
      // Promote, with the counters reset to no hits plus a reference of
      // ours, taken in the same step so the charge can be read.
      SlotMeta reset = meta;
      reset.SetAcquireCounter(1);
      reset.SetReleaseCounter(0);
      if (!h.meta.CasStrong(meta, reset)) {
        continue;
      }
      TEST_SYNC_POINT("FIFOCache::ProbationStep:Promoting");
      // Our reference keeps the slot's entry in place, but it may be a later
      // one with the same meta, in a slot emptied and refilled since the
      // word was read. Then the word no longer matches, LeaveProbation does
      // nothing, and the entry gets back the counts the reset took, up to
      // one more than the clock keeps, which reads them the same and can't
      // overflow however many hits it took meanwhile.
      *promoted = LeaveProbation(idx, word, h.total_charge,
                                 /*counters_reset=*/true);
      if (*promoted) {
        data->promoted_count += 1;
        if ((cur & kSeenPinned) == 0) {
          data->allowance_shrink_charge += h.total_charge;
        }
      } else {
        const uint32_t restored =
            std::min(acquire_count, uint32_t{ClockHandle::kMaxCountdown} + 1);
        SlotMeta before;
        h.meta.Apply(
            AcquireCounter::PlusTransformPromiseNoOverflow(restored) +
                ReleaseCounter::PlusTransformPromiseNoOverflow(restored),
            &before);
        CorrectNearOverflow(before, h.meta);
      }
      // Frees it if it was erased meanwhile and this was the last reference.
      Release(&h, /*useful=*/false, /*erase_if_last_ref=*/false);
      return true;
    }
    // Evict it (or free it, if it was erased while referenced).
    SlotMeta construction_meta;
    construction_meta.SetUnderConstruction();
    construction_meta.SetHit(meta.GetHit());
    if (!h.meta.CasStrong(meta, construction_meta)) {
      continue;
    }
    TEST_SYNC_POINT("FIFOCache::ProbationStep:Owned");
    // Took ownership, which keeps the slot's entry in place. It may be a
    // later one with the same meta (see above). Ownership can't be handed
    // back: while it is held, an Erase passes over the entry and a purge of
    // its chain unlinks it. So that entry goes instead, which any unpinned
    // entry may; the removal below goes by its own word.
    const bool queued_entry =
        (w.LoadRelaxed() & kProbationIdentity) == (word & kProbationIdentity);
    const size_t total_charge = h.total_charge;
    if (queued_entry && (cur & kSeenPinned) == 0) {
      data->allowance_shrink_charge += total_charge;
    }
    const bool from_probation = TrackEvictionRemoval(idx, total_charge, data);
    if (from_probation && meta.IsVisible() && !IsPlaceholder(h)) {
      // Evicted rather than erased, and not a placeholder. Stamped after the
      // call's earlier probation evictions, as one call per eviction would
      // stamp it.
      GhostPut(h.hashed_key, PendingProbationEvictions(*data));
    }
    data->freed_charge += total_charge;
    data->freed_count += 1;
    if (from_probation) {
      CountProbationEviction(data);
    }
    Remove(&h);
    TrackAndReleaseEvictedEntry(&h);
    return true;
  }
  // Kept losing races to hits: give it another lap, which doesn't make it a
  // pinned head's.
  if (!ProbationPush(idx, word, state, (v & kRotatedCell) != 0) &&
      LeaveProbationUnowned(idx, word)) {
    probation_full_.FetchAddRelaxed(1);
  }
  return true;
}

bool AutoHyperClockTable::LeaveProbationUnowned(size_t idx, uint32_t word) {
  bool moved = false;
  WithReference(idx, [&](HandleImpl& h) {
    moved = LeaveProbation(idx, word, h.total_charge);
  });
  return moved;
}

void AutoHyperClockTable::CountEvictions(const EvictionData& data) {
  // The insert's probation charge and the removals in one change.
  if (data.probation_precharge > data.probation_removed_charge) {
    probation_usage_.FetchAddRelaxed(data.probation_precharge -
                                     data.probation_removed_charge);
  } else if (data.probation_removed_charge > data.probation_precharge) {
    probation_usage_.FetchSubRelaxed(data.probation_removed_charge -
                                     data.probation_precharge);
  }
  if (data.probation_pinned_removed_charge > 0) {
    probation_pinned_usage_.FetchSubRelaxed(
        data.probation_pinned_removed_charge);
  }
  if (data.resident_removed > 0) {
    resident_count_.FetchSubRelaxed(
        static_cast<int64_t>(data.resident_removed));
  }
  if (data.allowance_grow_charge > 0 || data.allowance_shrink_charge > 0) {
    // Rounded up, so small charges give some back too.
    const size_t give_back =
        data.allowance_shrink_charge / kPinnedAllowanceDecay +
        (data.allowance_shrink_charge % kPinnedAllowanceDecay != 0 ? 1 : 0);
    const size_t cap = kPinnedAllowanceTargets * ProbationTarget();
    size_t cur = pinned_allowance_.LoadRelaxed();
    for (;;) {
      size_t next = std::min(cur, cap);
      if (data.allowance_grow_charge >= give_back) {
        next += std::min(data.allowance_grow_charge - give_back, cap - next);
      } else {
        next -= std::min(give_back - data.allowance_grow_charge, next);
      }
      if (next == cur || pinned_allowance_.CasWeakRelaxed(cur, next)) {
        break;
      }
    }
  }
  if (data.promoted_count > 0) {
    promotions_.FetchAddRelaxed(data.promoted_count);
  }
  if (PendingProbationEvictions(data) > 0) {
    probation_evictions_.FetchAddRelaxed(PendingProbationEvictions(data));
  }
  if (data.pinned_twice_count > 0) {
    pinned_twice_.FetchAddRelaxed(data.pinned_twice_count);
  }
  if (data.clock_rests > 0) {
    resident_all_pinned_.FetchAddRelaxed(data.clock_rests);
  }
  if (data.clock_rests_called_off > 0) {
    clock_rests_called_off_.FetchAddRelaxed(data.clock_rests_called_off);
  }
  if (data.freed_count > data.probation_freed_count) {
    resident_evictions_.FetchAddRelaxed(data.freed_count -
                                        data.probation_freed_count);
  }
}

void AutoHyperClockTable::CountProbationEviction(EvictionData* data) {
  ++data->probation_freed_count;
  const size_t pending = PendingProbationEvictions(*data);
  if (pending >= kMaxPendingProbationEvictions) {
    const uint64_t evictions =
        probation_evictions_.FetchAddRelaxed(pending) + pending;
    data->probation_freed_published = data->probation_freed_count;
    // A call can add more than 2^30 evictions before the next Evict starts.
    EndClockRestIfOver(clock_laps_.LoadRelaxed(), evictions);
  }
}

void AutoHyperClockTable::AddStats(FIFOCacheStats* stats) const {
  stats->probation_usage += GetProbationUsage();
  stats->pinned_allowance += PinnedAllowance(ProbationTarget());
  stats->promotions += promotions_.LoadRelaxed();
  stats->ghost_returns += ghost_returns_.LoadRelaxed();
  stats->probation_evictions += probation_evictions_.LoadRelaxed();
  stats->resident_evictions += resident_evictions_.LoadRelaxed();
  stats->effort_cap_stops += GetEvictionEffortExceededCount();
  stats->probation_full += probation_full_.LoadRelaxed();
  stats->pinned_twice += pinned_twice_.LoadRelaxed();
  // A rest is counted when its call returns, which can be after the call
  // that calls it off returns, so the difference is held at 0 or more.
  const uint64_t called_off = clock_rests_called_off_.LoadRelaxed();
  const uint64_t rests = resident_all_pinned_.LoadRelaxed();
  stats->resident_all_pinned += rests - std::min(rests, called_off);
}

size_t AutoHyperClockTable::CalcMaxUsableLength(size_t reserved_capacity) {
  // FIFOCache: one slot per 64 bytes, at least 2048, under either metadata
  // policy, as FIFOCache reserved before this table: at the load factor,
  // about one entry per 80 bytes. Slot indices fit the probation queue's
  // 32 bits, and the slots' bytes fit size_t (with 32 bits, it is the
  // smaller bound).
  constexpr size_t kMinSlots = 2048;
  constexpr size_t kBytesPerSlot = 64;
  constexpr uint64_t kMaxSlots =
      std::min(uint64_t{1} << 32, uint64_t{SIZE_MAX / sizeof(HandleImpl)});
  const size_t slots_per_page = kPresumedPageSize / sizeof(HandleImpl);
  size_t num_slots = std::max(kMinSlots, reserved_capacity / kBytesPerSlot);
  num_slots = static_cast<size_t>(
      std::min(uint64_t{num_slots}, kMaxSlots - slots_per_page));

  // Round up to page size
  return ((num_slots + slots_per_page - 1) / slots_per_page) * slots_per_page;
}

namespace {
bool IsHeadNonempty(const AutoHyperClockTable::HandleImpl& h) {
  return !h.head_next_with_shift.LoadRelaxed().IsEnd();
}
bool IsEntryAtHome(const AutoHyperClockTable::HandleImpl& h, int shift,
                   size_t home) {
  if (MatchAndRef(nullptr, nullptr, h, shift, home)) {
    Unref(h);
    return true;
  } else {
    return false;
  }
}
}  // namespace

void AutoHyperClockCache::ReportProblems(
    const std::shared_ptr<Logger>& info_log) const {
  BaseHyperClockCache::ReportProblems(info_log);

  if (info_log->GetInfoLogLevel() <= InfoLogLevel::DEBUG_LEVEL) {
    LoadVarianceStats head_stats;
    size_t entry_at_home_count = 0;
    uint64_t yield_count = 0;
    this->ForEachShard([&](const Shard* shard) {
      size_t count = shard->GetTableAddressCount();
      uint64_t length_info = UsedLengthToLengthInfo(count);
      for (size_t i = 0; i < count; ++i) {
        const auto& h = *shard->GetTable().HandlePtr(i);
        head_stats.Add(IsHeadNonempty(h));
        int shift;
        size_t home;
        GetHomeIndexAndShift(length_info, i, &home, &shift);
        assert(home == i);
        entry_at_home_count += IsEntryAtHome(h, shift, home);
      }
      yield_count += shard->GetTable().GetYieldCount();
    });
    ROCKS_LOG_AT_LEVEL(info_log, InfoLogLevel::DEBUG_LEVEL,
                       "Head occupancy stats: %s", head_stats.Report().c_str());
    ROCKS_LOG_AT_LEVEL(info_log, InfoLogLevel::DEBUG_LEVEL,
                       "Entries at home count: %zu", entry_at_home_count);
    ROCKS_LOG_AT_LEVEL(info_log, InfoLogLevel::DEBUG_LEVEL,
                       "Yield count: %" PRIu64, yield_count);
  }
}

// Upstream instantiates this through HyperClockCacheOptions::MakeSharedCache,
// which this copy leaves out.
template class BaseHyperClockCache<AutoHyperClockTable>;

}  // namespace fifo_cache

}  // namespace ROCKSDB_NAMESPACE
