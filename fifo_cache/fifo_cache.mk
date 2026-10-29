# RocksDB plugin definition. Link or copy this directory to
# <rocksdb>/plugin/fifo_cache and build RocksDB with ROCKSDB_PLUGINS=fifo_cache
# (make or CMake). RocksDB's CMake reads SOURCES and FUNC from here and the
# rest from CMakeLists.txt.
fifo_cache_SOURCES = fifo_cache.cc fifo_cache_plugin.cc
fifo_cache_HEADERS = fifo_cache_options.h
fifo_cache_FUNC = fifo_cache_register
fifo_cache_TESTS = fifo_cache_test.cc
