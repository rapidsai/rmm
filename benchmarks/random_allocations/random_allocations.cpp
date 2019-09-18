/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <rmm/mr/cnmem_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/sub_memory_resource.hpp>

#include <benchmark/benchmark.h>

namespace {

struct allocation {
  void* p{nullptr};
  std::size_t size{0};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};

void mixed_random_allocation_free(rmm::mr::device_memory_resource& mr,
                                  cudaStream_t stream = 0)
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  constexpr std::size_t size_mb{1 << 20};
  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(
    1, MAX_ALLOCATION_SIZE);

  constexpr int allocation_probability = 53; // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations-1);

  int active_allocations{0};
  int allocation_count{0};

  std::vector<allocation> allocations;

  for (int i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc = (chance < allocation_probability) && 
                 (allocation_count < num_allocations);
    }

    if (do_alloc) {
      size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      allocations.emplace_back(mr.allocate(size, stream), size);
      auto new_allocation = allocations.back();
    }
    else {
      size_t index = index_distribution(generator) % active_allocations;
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      mr.deallocate(to_free.p, to_free.size, stream);
    }
  }

  assert(active_allocations == allocations.size() == 0);
}
}  // namespace

static void BM_RandomAllocationsCUDA(benchmark::State& state) {
  rmm::mr::cuda_memory_resource mr;

  for (auto _ : state)
    mixed_random_allocation_free(mr);
}
// Register the function as a benchmark
BENCHMARK(BM_RandomAllocationsCUDA);

static void BM_RandomAllocationsCnmem(benchmark::State& state) {
  rmm::mr::cnmem_memory_resource mr;

  for (auto _ : state)
    mixed_random_allocation_free(mr);
}
BENCHMARK(BM_RandomAllocationsCnmem);

static void BM_RandomAllocationsSub(benchmark::State& state) {
  rmm::mr::sub_memory_resource mr;

  for (auto _ : state)
    mixed_random_allocation_free(mr);
}
BENCHMARK(BM_RandomAllocationsSub);
