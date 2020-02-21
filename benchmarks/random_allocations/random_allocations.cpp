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
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/sub_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>

#include <benchmark/benchmark.h>

#include <random>
#include <cstdlib>

#define VERBOSE 0

namespace {

constexpr std::size_t size_mb{1 << 20};

struct allocation {
  void* p{nullptr};
  std::size_t size{0};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};

using allocation_vector = std::vector<allocation>;

allocation remove_at(allocation_vector& allocs, std::size_t index) {
  assert(index < allocs.size());
  auto removed = allocs[index];

  if ((allocs.size() > 1) && (index < allocs.size() - 1)) {
    std::swap(allocs[index], allocs.back());
  }
  allocs.pop_back();

  return removed;
}

template <typename SizeDistribution>
void random_allocation_free(rmm::mr::device_memory_resource& mr,
                            SizeDistribution size_distribution,
                            size_t num_allocations,
                            size_t max_usage, // in MiB
                            cudaStream_t stream = 0)
{
  std::default_random_engine generator;

  max_usage *= size_mb; // convert to bytes
  
  constexpr int allocation_probability = 73; // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations-1);

  int active_allocations{0};
  int allocation_count{0};

  allocation_vector allocations{};
  size_t allocation_size{0};
  size_t total_allocated{0};

  for (int i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    size_t size = static_cast<size_t>(size_distribution(generator));
    
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc = (chance < allocation_probability) &&
                 (allocation_count < num_allocations) &&
                 (allocation_size + size < max_usage);
    }

    void* ptr = nullptr;
    if (do_alloc) { // try to allocate
      try {
        ptr = mr.allocate(size, stream);
      } catch(std::bad_alloc) {
        do_alloc = false;
        #if VERBOSE
          std::cout << "FAILED to allocate " << size << "\n";
        #endif
      }
    }

    if (do_alloc) { // alloc succeeded
      allocations.emplace_back(ptr, size);
      active_allocations++;
      allocation_count++;
      allocation_size += size;

      #if VERBOSE
      std::cout << active_allocations << " | " << allocation_count << " Allocating: " << size 
                << " | total: " << allocation_size << "\n";
      #endif
    }
    else { // dealloc, or alloc failed
      if (active_allocations > 0) {
        size_t index = index_distribution(generator) % active_allocations;
        active_allocations--;
        allocation to_free = remove_at(allocations, index);
        mr.deallocate(to_free.p, to_free.size, stream);
        allocation_size -= to_free.size;

        #if VERBOSE
        std::cout << active_allocations << " | " << allocation_count << " Deallocating: " 
                  << to_free.size << " at " << index << " | total: " << allocation_size << "\n";
        #endif
      }
    }
  }

  //std::cout << "TOTAL ALLOCATIONS: " << allocation_count << "\n";

  assert(active_allocations == 0);
  assert(allocations.size() == 0);
}
}  // namespace

void uniform_random_allocations(rmm::mr::device_memory_resource& mr,
                                size_t num_allocations,
                                size_t max_allocation_size, // in MiB
                                size_t max_usage,
                                cudaStream_t stream = 0) {
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_allocation_size * size_mb);
  random_allocation_free(mr, size_distribution, num_allocations, max_usage, stream);
}

// TODO figure out how to map a normal distribution to integers between 1 and max_allocation_size
/*void normal_random_allocations(rmm::mr::device_memory_resource& mr,
                                size_t num_allocations = 1000,
                                size_t mean_allocation_size = 500, // in MiB
                                size_t stddev_allocation_size = 500, // in MiB
                                size_t max_usage = 8 << 20,
                                cudaStream_t stream) {
  std::normal_distribution<std::size_t> size_distribution(, max_allocation_size * size_mb);
}*/

constexpr size_t max_usage = 16000;

template <typename MemoryResource>
static void BM_RandomAllocations(benchmark::State& state) {
  MemoryResource mr;

  size_t num_allocations = state.range(0);
  size_t max_size = state.range(1);

  try {
    for (auto _ : state)
      uniform_random_allocations(mr, num_allocations, max_size, max_usage);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}

static void num_range(benchmark::internal::Benchmark* b, int size) {
  for (int num_allocations : std::vector<int>{1000, 10000, 100000})
    b->Args({num_allocations, size})->Unit(benchmark::kMillisecond);
}


static void size_range(benchmark::internal::Benchmark* b, int num) {
  for (int max_size : std::vector<int>{1, 4, 64, 256, 1024, 4096})
    b->Args({num, max_size})->Unit(benchmark::kMillisecond);
}

static void num_size_range(benchmark::internal::Benchmark* b) {
  for (int num_allocations : std::vector<int>{1000, 10000, 100000})
    size_range(b, num_allocations);
}


int num_allocations = -1;
int max_size = -1;

static void benchmark_range(benchmark::internal::Benchmark* b) {
  if (num_allocations > 0) {
    if (max_size > 0)
      b->Args({num_allocations, max_size})->Unit(benchmark::kMillisecond);
    else
      size_range(b, num_allocations);
  } else {
    if (max_size > 0)
      num_range(b, max_size);
    else
      num_size_range(b);
  }
}

void declare_benchmark(std::string name) {
  if (name == "cuda")
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::cuda_memory_resource)->Apply(benchmark_range);
  if (name == "hybrid")
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::hybrid_memory_resource)->Apply(benchmark_range);
  else if (name == "sub")
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::sub_memory_resource)->Apply(benchmark_range);
  else if (name == "cnmem")
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::cnmem_memory_resource)->Apply(benchmark_range);
  else
    std::cout << "Error: invalid memory_resource name: " << name << "\n";
}

int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  if (argc > 1) {
    std::string mr_name = argv[1];
    if (argc > 2) num_allocations = atoi(argv[2]);
    if (argc > 3) max_size = atoi(argv[3]);
    declare_benchmark(mr_name);
  }
  else {
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::sub_memory_resource)->Apply(benchmark_range);
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::hybrid_memory_resource)->Apply(benchmark_range);
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::cnmem_memory_resource)->Apply(benchmark_range);
    BENCHMARK_TEMPLATE(BM_RandomAllocations, rmm::mr::cuda_memory_resource)->Apply(benchmark_range);
  }

  ::benchmark::RunSpecifiedBenchmarks();
}

// For profiling
/*template <typename MemoryResource>
static void RandomAllocations(size_t num_allocations, size_t max_size) {
  MemoryResource mr;

  try {
    uniform_random_allocations(mr, num_allocations, max_size, max_usage);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}

int main(int argc, char** argv) {
  std::string name{ argc > 1 ? argv[1] : "cnmem" };

  if (name == "hybrid")
    RandomAllocations<rmm::mr::hybrid_memory_resource>(1000, 4096);
  else if (name == "sub")
    RandomAllocations<rmm::mr::sub_memory_resource>(1000, 4096);
  else if (name == "cnmem")
    RandomAllocations<rmm::mr::cnmem_memory_resource>(1000, 4096);

  return 0;
}*/



