# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under both the GPLv2 (found in the
# COPYING file in the root directory) and Apache 2.0 License
# (found in the LICENSE.Apache file in the root directory).

# Read by RocksDB's CMake when it is configured with ROCKSDB_PLUGINS=fifo_cache
# (see fifo_cache.mk, from which it takes the sources and the registration
# function). This directory is not a standalone CMake project.
set(fifo_cache_HEADERS fifo_cache_options.h PARENT_SCOPE)
set(fifo_cache_TESTS fifo_cache_test.cc PARENT_SCOPE)
