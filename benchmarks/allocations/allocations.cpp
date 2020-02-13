/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <benchmark/benchmark.h>

#include <chrono>
#include <iostream>
#include <thread>

struct allocation {
  void* p{};
  std::size_t size{};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};

/**
 * @brief Performs a set of allocations of sizes (in bytes) determined by the
 * range `[sizes_begin, sizes_end)` and then frees all of the allocations.
 *
 * Requires a pre-allocated `scratch` vector of at least size
 * `std::distance(sizes_begin, sizes_end)` to track each allocation. This is to
 * eliminate the overhead of allocating the scratch vector from this benchmark.
 *
 * @tparam SizeIterator Input iterator
 * @param mr Resource to use for allocation/free
 * @param sizes_begin Start of range of allocation sizes
 * @param sizes_end End of range of allocation sizes
 * @param scratch Preallocated vector of `allocation`s of at least size
 * `std::distance(sizes_begin, sizes_end)` used for tracking each allocation.
 */
template <typename SizeIterator>
void bulk_alloc_free(rmm::mr::device_memory_resource* mr,
                     SizeIterator sizes_begin, SizeIterator sizes_end,
                     std::vector<allocation>& scratch) {
  // Do all allocations
  std::transform(
      sizes_begin, sizes_end, scratch.begin(),
      [mr](std::size_t allocation_size) {
        return allocation{mr->allocate(allocation_size), allocation_size};
      });

  // Free all allocations
  std::for_each(scratch.begin(), scratch.end(),
                [mr](allocation a) { mr->deallocate(a.p, a.size); });
}

static void BM_test(benchmark::State& state) {
  rmm::mr::cuda_memory_resource mr;

  std::vector<std::size_t> sizes(100);
  std::iota(sizes.begin(), sizes.end(), 1);
  std::vector<allocation> allocation_scratch(sizes.size());

  try {
    for (auto _ : state) {
      bulk_alloc_free(&mr, sizes.begin(), sizes.end(), allocation_scratch);
    }
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}
BENCHMARK(BM_test)->Unit(benchmark::kMillisecond);