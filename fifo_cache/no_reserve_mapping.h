//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <cstddef>

#include "port/mmap.h"

namespace ROCKSDB_NAMESPACE {
namespace fifo_cache {

// The slot table's memory: like TypedMemMapping<T> from
// MemMapping::AllocateLazyZeroed, but on Linux the mapping does not count
// against the commit limit (MAP_NORESERVE; Linux ignores it under
// vm.overcommit_memory=2), so a large reservation of which little is
// touched does not inflate committed memory. As with any overcommitted
// memory, touching a page when memory is exhausted fails at that point.
// Elsewhere the same as AllocateLazyZeroed. MemMapping has no such
// allocation in the RocksDB releases this plugin builds against.
template <typename T>
class NoReserveTypedMemMapping {
 public:
#if !defined(OS_WIN) && defined(MAP_NORESERVE)
  explicit NoReserveTypedMemMapping(size_t length) : length_(length) {
    if (length > 0) {
      addr_ = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (addr_ == MAP_FAILED) {
        addr_ = nullptr;
      }
    }
  }
  ~NoReserveTypedMemMapping() {
    if (addr_ != nullptr) {
      munmap(addr_, length_);
    }
  }

  // nullptr on failure
  T* Get() const { return static_cast<T*>(addr_); }

 private:
  void* addr_ = nullptr;
  size_t length_;
#else
  explicit NoReserveTypedMemMapping(size_t length)
      : mapping_(MemMapping::AllocateLazyZeroed(length)) {}

  // nullptr on failure
  T* Get() const { return mapping_.Get(); }

 private:
  TypedMemMapping<T> mapping_;
#endif

 public:
  NoReserveTypedMemMapping(const NoReserveTypedMemMapping&) = delete;
  NoReserveTypedMemMapping& operator=(const NoReserveTypedMemMapping&) = delete;
};

}  // namespace fifo_cache
}  // namespace ROCKSDB_NAMESPACE
