/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/cxxopts.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>

#define VERBOSE 0

namespace {
constexpr std::size_t size_mb{1 << 20};

struct allocation {
  void* ptr{nullptr};
  std::size_t size{0};
  allocation(void* ptr, std::size_t size) : ptr{ptr}, size{size} {}
  allocation() = default;
};

using allocation_vector = std::vector<allocation>;

allocation remove_at(allocation_vector& allocs, std::size_t index)
{
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
                            std::size_t num_allocations,
                            std::size_t max_usage,  // in MiB
                            rmm::cuda_stream_view stream = {})
{
  std::default_random_engine generator;

  max_usage *= size_mb;  // convert to bytes

  constexpr int allocation_probability{73};  // percent
  constexpr int max_op_chance{99};
  std::uniform_int_distribution<int> op_distribution(0, max_op_chance);
  std::uniform_int_distribution<int> index_distribution(0, static_cast<int>(num_allocations) - 1);

  int active_allocations{0};
  std::size_t allocation_count{0};

  allocation_vector allocations{};
  std::size_t allocation_size{0};

  for (std::size_t i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    auto size     = static_cast<std::size_t>(size_distribution(generator));

    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc   = (chance < allocation_probability) && (allocation_count < num_allocations) &&
                 (allocation_size + size < max_usage);
    }

    void* ptr = nullptr;
    if (do_alloc) {  // try to allocate
      try {
        ptr = mr.allocate(size, stream);
      } catch (rmm::bad_alloc const&) {
        do_alloc = false;
#if VERBOSE
        std::cout << "FAILED to allocate " << size << "\n";
#endif
      }
    }

    if (do_alloc) {  // alloc succeeded
      allocations.emplace_back(ptr, size);
      active_allocations++;
      allocation_count++;
      allocation_size += size;

#if VERBOSE
      std::cout << active_allocations << " | " << allocation_count << " Allocating: " << size
                << " | total: " << allocation_size << "\n";
#endif
    } else {  // dealloc, or alloc failed
      if (active_allocations > 0) {
        std::size_t index = index_distribution(generator) % active_allocations;
        active_allocations--;
        allocation to_free = remove_at(allocations, index);
        mr.deallocate(to_free.ptr, to_free.size, stream);
        allocation_size -= to_free.size;

#if VERBOSE
        std::cout << active_allocations << " | " << allocation_count
                  << " Deallocating: " << to_free.size << " at " << index
                  << " | total: " << allocation_size << "\n";
#endif
      }
    }
  }

  // std::cout << "TOTAL ALLOCATIONS: " << allocation_count << "\n";

  assert(active_allocations == 0);
  assert(allocations.size() == 0);
}
}  // namespace

void uniform_random_allocations(
  rmm::mr::device_memory_resource& mr,
  std::size_t num_allocations,      // NOLINT(bugprone-easily-swappable-parameters)
  std::size_t max_allocation_size,  // size in MiB
  std::size_t max_usage,
  rmm::cuda_stream_view stream = {})
{
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_allocation_size * size_mb);
  random_allocation_free(mr, size_distribution, num_allocations, max_usage, stream);
}

// TODO figure out how to map a normal distribution to integers between 1 and max_allocation_size
/*void normal_random_allocations(rmm::mr::device_memory_resource& mr,
                                std::size_t num_allocations = 1000,
                                std::size_t mean_allocation_size = 500, // in MiB
                                std::size_t stddev_allocation_size = 500, // in MiB
                                std::size_t max_usage = 8 << 20,
                                cuda_stream_view stream) {
  std::normal_distribution<std::size_t> size_distribution(, max_allocation_size * size_mb);
}*/

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_cuda_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda(), rmm::percent_of_free_device_memory(50));
}

inline auto make_arena()
{
  auto free = rmm::available_device_memory().first;
  constexpr auto reserve{64UL << 20};  // Leave some space for CUDA overhead.
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda(), free - reserve);
}

inline auto make_binning()
{
  auto pool = make_pool();
  // Add a binning_memory_resource with fixed-size bins of sizes 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  constexpr auto min_bin_pow2{18};
  constexpr auto max_bin_pow2{22};
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(
    pool, min_bin_pow2, max_bin_pow2);
  return mr;
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

constexpr std::size_t max_usage = 16000;

static void BM_RandomAllocations(benchmark::State& state, MRFactoryFunc const& factory)
{
  auto mr = factory();

  std::size_t num_allocations = state.range(0);
  std::size_t max_size        = state.range(1);

  try {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
      uniform_random_allocations(*mr, num_allocations, max_size, max_usage);
    }
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}

static void num_range(benchmark::internal::Benchmark* bench, int size)
{
  for (int num_allocations : std::vector<int>{1000, 10000, 100000}) {
    bench->Args({num_allocations, size})->Unit(benchmark::kMillisecond);
  }
}

static void size_range(benchmark::internal::Benchmark* bench, int num)
{
  for (int max_size : std::vector<int>{1, 4, 64, 256, 1024, 4096}) {
    bench->Args({num, max_size})->Unit(benchmark::kMillisecond);
  }
}

static void num_size_range(benchmark::internal::Benchmark* bench)
{
  for (int num_allocations : std::vector<int>{1000, 10000, 100000}) {
    size_range(bench, num_allocations);
  }
}

int num_allocations = -1;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
int max_size        = -1;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void benchmark_range(benchmark::internal::Benchmark* bench)
{
  if (num_allocations > 0) {
    if (max_size > 0) {
      bench->Args({num_allocations, max_size})->Unit(benchmark::kMillisecond);
    } else {
      size_range(bench, num_allocations);
    }
  } else {
    if (max_size > 0) {
      num_range(bench, max_size);
    } else {
      num_size_range(bench);
    }
  }
}

void declare_benchmark(std::string const& name)
{
  if (name == "cuda") {
    BENCHMARK_CAPTURE(BM_RandomAllocations, cuda_mr, &make_cuda)  // NOLINT
      ->Apply(benchmark_range);
  }
  if (name == "cuda_async") {
    BENCHMARK_CAPTURE(BM_RandomAllocations, cuda_async_mr, &make_cuda_async)  // NOLINT
      ->Apply(benchmark_range);
  } else if (name == "binning") {
    BENCHMARK_CAPTURE(BM_RandomAllocations, binning_mr, &make_binning)  // NOLINT
      ->Apply(benchmark_range);
  } else if (name == "pool") {
    BENCHMARK_CAPTURE(BM_RandomAllocations, pool_mr, &make_pool)  // NOLINT
      ->Apply(benchmark_range);
  } else if (name == "arena") {
    BENCHMARK_CAPTURE(BM_RandomAllocations, arena_mr, &make_arena)  // NOLINT
      ->Apply(benchmark_range);
  } else {
    std::cout << "Error: invalid memory_resource name: " << name << "\n";
  }
}

static void profile_random_allocations(MRFactoryFunc const& factory,
                                       std::size_t num_allocations,
                                       std::size_t max_size)
{
  auto mr = factory();

  try {
    uniform_random_allocations(*mr, num_allocations, max_size, max_usage);
  } catch (std::exception const& e) {
    std::cout << "Error: " << e.what() << "\n";
  }
}

int main(int argc, char** argv)
{
  try {
    // benchmark::Initialize will remove GBench command line arguments it
    // recognizes and leave any remaining arguments
    ::benchmark::Initialize(&argc, argv);

    // Parse for replay arguments:
    cxxopts::Options options("RMM Random Allocations Benchmark",
                             "Benchmarks random allocations within a size range.");

    options.add_options()(
      "p,profile", "Profiling mode: run once", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("r,resource",
                          "Type of device_memory_resource",
                          cxxopts::value<std::string>()->default_value("pool"));
    options.add_options()("n,numallocs",
                          "Number of allocations (default of 0 tests a range)",
                          cxxopts::value<int>()->default_value("1000"));
    options.add_options()("m,maxsize",
                          "Maximum allocation size (default of 0 tests a range)",
                          cxxopts::value<int>()->default_value("4096"));

    auto args       = options.parse(argc, argv);
    num_allocations = args["numallocs"].as<int>();
    max_size        = args["maxsize"].as<int>();

    if (args.count("profile") > 0) {
      std::map<std::string, MRFactoryFunc> const funcs({{"arena", &make_arena},
                                                        {"binning", &make_binning},
                                                        {"cuda", &make_cuda},
                                                        {"cuda_async", &make_cuda_async},
                                                        {"pool", &make_pool}});
      auto resource = args["resource"].as<std::string>();

      std::cout << "Profiling " << resource << " with " << num_allocations << " allocations of max "
                << max_size << "B\n";

      profile_random_allocations(funcs.at(resource), num_allocations, max_size);

      std::cout << "Finished\n";
    } else {
      if (args.count("numallocs") == 0) {  // if zero reset to -1 so we benchmark over a range
        num_allocations = -1;
      }
      if (args.count("maxsize") == 0) {  // if zero reset to -1 so we benchmark over a range
        max_size = -1;
      }

      if (args.count("resource") > 0) {
        std::string mr_name = args["resource"].as<std::string>();
        declare_benchmark(mr_name);
      } else {
        std::vector<std::string> mrs{"pool", "binning", "arena", "cuda_async", "cuda"};
        std::for_each(
          std::cbegin(mrs), std::cend(mrs), [](auto const& mr) { declare_benchmark(mr); });
      }
      ::benchmark::RunSpecifiedBenchmarks();
    }

  } catch (std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
