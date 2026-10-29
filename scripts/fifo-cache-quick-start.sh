#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under both the GPLv2 (found in the
# COPYING file in the root directory) and Apache 2.0 License
# (found in the LICENSE.Apache file in the root directory).

# Quick start for the FIFOCache plugin: builds RocksDB from source with this
# checkout's fifo_cache directory as a plugin (CMake, Release), installs it,
# and runs a short db_bench comparison of FIFOCache, LRUCache and
# HyperClockCache on one skewed read workload.
#
# Usage: scripts/fifo-cache-quick-start.sh [work-dir]
#   work-dir  New or empty directory for the RocksDB tree, the install prefix
#             and the benchmark DB. Default: build-fifo-cache-quick-start in
#             this checkout (ignored by git).
#
# Environment:
#   ROCKSDB_REVISION  RocksDB commit or tag to build. Default:
#                     LOCKED_ROCKSDB_REVISION from dependencies.lock.
#   BUILD_JOBS        Parallel build jobs (default: online CPUs).
#   RUN_BENCHMARK     1 to run the comparison (default), 0 to only build.
#   NUM_KEYS          Keys in the benchmark DB (default: 1000000, 400-byte
#                     values, about 400 MB).
#   CACHE_SIZE        Block cache capacity in bytes (default: 67108864).
#   THREADS           Reader threads (default: 8).
#   DURATION          Seconds per cache (default: 10).
#   CC, CXX           Compilers, as for any CMake build.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPOSITORY_ROOT

# shellcheck disable=SC1091 # Path is fixed relative to this script.
source "${REPOSITORY_ROOT}/dependencies.lock"

rocksdb_revision="${ROCKSDB_REVISION:-${LOCKED_ROCKSDB_REVISION:-}}"
if [[ -z "${rocksdb_revision}" ]]; then
  echo "Set ROCKSDB_REVISION or LOCKED_ROCKSDB_REVISION in dependencies.lock." >&2
  exit 1
fi
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2\n')}"
run_benchmark="${RUN_BENCHMARK:-1}"
num_keys="${NUM_KEYS:-1000000}"
cache_size="${CACHE_SIZE:-67108864}"
threads="${THREADS:-8}"
duration="${DURATION:-10}"

work="${1:-${REPOSITORY_ROOT}/build-fifo-cache-quick-start}"
mkdir -p "${work}"
work="$(cd -- "${work}" && pwd)"
if [[ -n "$(ls -A "${work}")" ]]; then
  echo "${work} is not empty; pass a new or empty directory." >&2
  exit 1
fi
rocksdb_source="${work}/rocksdb"
prefix="${work}/prefix"
db_bench="${rocksdb_source}/build/db_bench"

echo "RocksDB revision: ${rocksdb_revision} (${LOCKED_ROCKSDB_VERSION:-})"
echo "Work directory:   ${work}"

# RocksDB at the revision, with this checkout's fifo_cache as
# plugin/fifo_cache.
mkdir -p "${rocksdb_source}"
curl -fsSL --retry 3 \
  "https://github.com/facebook/rocksdb/archive/${rocksdb_revision}.tar.gz" |
  tar -xz --strip-components=1 -C "${rocksdb_source}"
ln -s "${REPOSITORY_ROOT}/fifo_cache" "${rocksdb_source}/plugin/fifo_cache"

cmake -S "${rocksdb_source}" -B "${rocksdb_source}/build" \
  -DCMAKE_BUILD_TYPE=Release -DROCKSDB_PLUGINS=fifo_cache
cmake --build "${rocksdb_source}/build" --parallel "${build_jobs}"
cmake --install "${rocksdb_source}/build" --prefix "${prefix}"

echo
echo "Built RocksDB with FIFOCache."
echo "  db_bench: ${db_bench}"
echo "  Prefix:   ${prefix}"
echo "  Header:   ${prefix}/include/rocksdb/plugin/fifo_cache/fifo_cache_options.h"

if [[ "${run_benchmark}" != 1 ]]; then
  exit 0
fi

# One DB, then the same skewed readrandom with each cache. Only the cache
# flags differ between runs.
db="${work}/bench_db"
common_args=(
  --db="${db}"
  --num="${num_keys}"
  --value_size=400
  --compression_type=none
)
echo
echo "Loading ${num_keys} keys into ${db}"
"${db_bench}" "${common_args[@]}" --benchmarks=fillrandom --disable_wal=1 \
  >"${work}/fill.log" 2>&1

results=()
for cache in FIFOCache LRUCache HyperClockCache; do
  case "${cache}" in
    FIFOCache) cache_args=(--cache_uri="fifo_cache://capacity=${cache_size}") ;;
    LRUCache) cache_args=(--cache_type=lru_cache --cache_size="${cache_size}") ;;
    HyperClockCache)
      cache_args=(--cache_type=hyper_clock_cache --cache_size="${cache_size}")
      ;;
  esac
  log="${work}/bench-${cache}.log"
  echo "Benchmarking ${cache} (${threads} threads, ${duration}s)"
  "${db_bench}" "${common_args[@]}" "${cache_args[@]}" \
    --use_existing_db=1 --benchmarks=readrandom --seed=1 \
    --read_random_exp_range=10 --threads="${threads}" \
    --duration="${duration}" --statistics >"${log}" 2>&1
  if [[ "${cache}" == FIFOCache ]] &&
    ! grep -q 'block_cache_name: FIFOCache' "${db}/LOG"; then
    echo "db_bench did not use FIFOCache; see ${db}/LOG." >&2
    exit 1
  fi
  ops="$(awk '/^readrandom /{for (i = 1; i <= NF; i++) if ($i == "ops/sec") print $(i - 1)}' "${log}")"
  miss_ratio="$(awk '
    /rocksdb.block.cache.data.miss COUNT/ {miss = $NF}
    /rocksdb.block.cache.data.hit COUNT/ {hit = $NF}
    END {if (miss + hit > 0) printf "%.3f", miss / (miss + hit)}' "${log}")"
  results+=("$(printf '%-16s %12s %18s' "${cache}" "${ops}" "${miss_ratio}")")
done

echo
echo "readrandom, ${num_keys} keys, $((cache_size >> 20)) MiB cache, ${threads} threads, ${duration}s each:"
printf '%-16s %12s %18s\n' cache ops/sec data-miss-ratio
printf '%s\n' "${results[@]}"
echo
echo "Logs: ${work}/bench-*.log. Short runs on a busy machine are noisy;"
echo "compare caches on an otherwise idle host and repeat for firm numbers."
