#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under both the GPLv2 (found in the
# COPYING file in the root directory) and Apache 2.0 License
# (found in the LICENSE.Apache file in the root directory).

# Builds RocksDB from source with the FIFOCache plugin (fifo_cache/), warnings
# as errors, and runs the plugin's tests:
#   debug:   CMake, CMAKE_BUILD_TYPE=Debug (assertions and sync points).
#   release: make, DEBUG_LEVEL=0 (RocksDB's optimized release flags), then a
#            db_bench run that selects the cache with --cache_uri only.
#
# Environment:
#   ROCKSDB_REVISION           RocksDB commit or tag to build. Default:
#                              LOCKED_ROCKSDB_REVISION from dependencies.lock.
#   FIFO_CACHE_PLUGIN_BUILDS   Builds to run (default: "debug release").
#   BUILD_DIRECTORY            Work directory (default: build-fifo-cache-plugin).
#   BUILD_JOBS                 Parallel jobs (default: online CPUs).
#   CC, CXX                    Compilers, as for any CMake or make build.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPOSITORY_ROOT

dependency_lock_file="${DEPENDENCY_LOCK_FILE:-${REPOSITORY_ROOT}/dependencies.lock}"
# shellcheck disable=SC1090,SC1091 # The lock path can be overridden.
source "${dependency_lock_file}"

rocksdb_revision="${ROCKSDB_REVISION:-${LOCKED_ROCKSDB_REVISION:-}}"
if [[ -z "${rocksdb_revision}" ]]; then
  echo "Set ROCKSDB_REVISION or LOCKED_ROCKSDB_REVISION in ${dependency_lock_file}." >&2
  exit 1
fi
builds="${FIFO_CACHE_PLUGIN_BUILDS:-debug release}"
build_directory="${BUILD_DIRECTORY:-${REPOSITORY_ROOT}/build-fifo-cache-plugin}"
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2\n')}"
rocksdb_source="${build_directory}/rocksdb"

echo "RocksDB revision: ${rocksdb_revision}"
echo "Builds:           ${builds}"
echo "Build directory:  ${build_directory}"
echo "Build jobs:       ${build_jobs}"
echo "Compilers:        CC=${CC:-default} CXX=${CXX:-default}"

# A RocksDB source tree at the revision, with a copy of this repository's
# fifo_cache directory as plugin/fifo_cache. (A copy, because make writes its
# objects next to the sources.)
revision_marker="${rocksdb_source}/.rocksdb-revision"
if [[ ! -f "${revision_marker}" ||
  "$(cat "${revision_marker}")" != "${rocksdb_revision}" ]]; then
  rm -rf "${rocksdb_source}"
  mkdir -p "${rocksdb_source}"
  curl -fsSL --retry 3 \
    "https://github.com/facebook/rocksdb/archive/${rocksdb_revision}.tar.gz" |
    tar -xz --strip-components=1 -C "${rocksdb_source}"
  printf '%s\n' "${rocksdb_revision}" > "${revision_marker}"
fi
rm -rf "${rocksdb_source}/plugin/fifo_cache"
cp -Rp "${REPOSITORY_ROOT}/fifo_cache" "${rocksdb_source}/plugin/fifo_cache"

compiler_launcher=""
if command -v ccache >/dev/null 2>&1; then
  compiler_launcher="$(command -v ccache)"
fi

for build in ${builds}; do
  case "${build}" in
    debug)
      echo "::group::FIFOCache plugin: CMake Debug build"
      cmake_args=(
        -S "${rocksdb_source}"
        -B "${build_directory}/debug"
        -DCMAKE_BUILD_TYPE=Debug
        -DROCKSDB_PLUGINS=fifo_cache
        -DFAIL_ON_WARNINGS=ON
        -DWITH_GFLAGS=OFF
      )
      if command -v ninja >/dev/null 2>&1; then
        cmake_args+=(-G Ninja)
      fi
      if [[ -n "${compiler_launcher}" ]]; then
        cmake_args+=(
          -DCMAKE_C_COMPILER_LAUNCHER="${compiler_launcher}"
          -DCMAKE_CXX_COMPILER_LAUNCHER="${compiler_launcher}"
        )
      fi
      cmake "${cmake_args[@]}"
      cmake --build "${build_directory}/debug" --target fifo_cache_test \
        --parallel "${build_jobs}"
      echo "::endgroup::"
      "${build_directory}/debug/fifo_cache_test"
      ;;
    release)
      echo "::group::FIFOCache plugin: make DEBUG_LEVEL=0 build"
      # Link the test with only RocksDB's test harness: the full test library
      # (db_test_util) does not compile with DEBUG_LEVEL=0. Make expands it.
      # shellcheck disable=SC2016
      test_library='$(TESTHARNESS)'
      make_args=(
        -C "${rocksdb_source}"
        -j "${build_jobs}"
        DEBUG_LEVEL=0
        # The tests use dynamic_cast; the library itself does not need RTTI.
        USE_RTTI=1
        ROCKSDB_PLUGINS=fifo_cache
        "TEST_LIBRARY=${test_library}"
      )
      if [[ -n "${compiler_launcher}" ]]; then
        make_args+=(
          CC="${compiler_launcher} ${CC:-cc}"
          CXX="${compiler_launcher} ${CXX:-c++}"
        )
      fi
      make "${make_args[@]}" fifo_cache_test db_bench
      echo "::endgroup::"
      "${rocksdb_source}/fifo_cache_test"

      # db_bench names no FIFOCache; the plugin's registration makes
      # --cache_uri create one, which the DB's LOG reports.
      db_directory="${build_directory}/db_bench_db"
      rm -rf "${db_directory}"
      "${rocksdb_source}/db_bench" \
        --db="${db_directory}" \
        --benchmarks=fillseq,readrandom \
        --num=100000 \
        --reads=100000 \
        --cache_uri='fifo_cache://capacity=8388608;num_shard_bits=2' \
        --statistics
      grep -q 'block_cache_name: FIFOCache' "${db_directory}/LOG"
      grep 'block_cache_name: \|capacity : ' "${db_directory}/LOG" | head -n 2
      if "${rocksdb_source}/db_bench" --db="${db_directory}" \
        --benchmarks=readrandom --use_existing_db=1 --reads=1 \
        --cache_uri='fifo_cache://capacity=0' >/dev/null 2>&1; then
        echo "db_bench accepted fifo_cache://capacity=0" >&2
        exit 1
      fi
      echo "db_bench used FIFOCache from --cache_uri and rejected a bad one."
      ;;
    *)
      echo "Unknown build '${build}' in FIFO_CACHE_PLUGIN_BUILDS." >&2
      exit 1
      ;;
  esac
done
