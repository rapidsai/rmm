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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
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
                                  size_t num_allocations = 10000,
                                  size_t max_allocation_size = 500, // in MiB
                                  cudaStream_t stream = 0)
{
  std::default_random_engine generator;

  constexpr std::size_t size_mb{1 << 20};
  std::uniform_int_distribution<std::size_t> size_distribution(
    1, max_allocation_size * size_mb);

  constexpr int allocation_probability = 53; // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations-1);

  int active_allocations{0};
  int allocation_count{0};

  std::vector<allocation> allocations;
  size_t allocation_size{0};

  size_t total_mem, free_mem;
  cudaError_t res = cudaMemGetInfo(&free_mem, &total_mem);
  assert(res == cudaSuccess);

  const size_t MB{1 << 20};
  const size_t MEM_USAGE_PERCENTAGE{8870};
  size_t max_size = (((free_mem / MB) * MEM_USAGE_PERCENTAGE) / 10000) * MB;

  for (int i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    size_t size = size_distribution(generator);
    
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc = (chance < allocation_probability) &&
                 (allocation_count < num_allocations) &&
                 (allocation_size + size < max_size);
    }

    if (do_alloc) {
      active_allocations++;
      allocation_count++;
      allocations.emplace_back(mr.allocate(size, stream), size);
      allocation_size += size;
    }
    else {
      size_t index = index_distribution(generator) % active_allocations;
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      mr.deallocate(to_free.p, to_free.size, stream);
      allocation_size -= to_free.size;
    }
  }

  assert(active_allocations == 0);
  assert(allocations.size() == 0);
}
}  // namespace

static void BM_RandomAllocationsCUDA(benchmark::State& state) {
  rmm::mr::cuda_memory_resource mr;

  try {
    for (auto _ : state)
      mixed_random_allocation_free(mr);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}
//BENCHMARK(BM_RandomAllocationsCUDA)->Unit(benchmark::kMillisecond);

template <typename State>
static void BM_RandomAllocationsSub(State& state) {
  rmm::mr::sub_memory_resource mr;

  try {
    for (auto _ : state)
      mixed_random_allocation_free(mr);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}
BENCHMARK(BM_RandomAllocationsSub)->Unit(benchmark::kMillisecond);

template <typename State>
static void BM_RandomAllocationsCnmem(State& state) {
  rmm::mr::cnmem_memory_resource mr;

  try {
    for (auto _ : state)
      mixed_random_allocation_free(mr);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}
//BENCHMARK(BM_RandomAllocationsCnmem)->Unit(benchmark::kMillisecond);

/*int main(void) {
  std::vector<int> state(1000);
  BM_RandomAllocationsSub(state);
  return 0;
}*/



