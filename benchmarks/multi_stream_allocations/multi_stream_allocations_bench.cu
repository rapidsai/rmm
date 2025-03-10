/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/cxxopts.hpp>

#include <cstddef>

__global__ void compute_bound_kernel(int64_t* out)
{
  clock_t clock_begin   = clock64();
  clock_t clock_current = clock_begin;
  auto const million{1'000'000};

  if (threadIdx.x == 0) {  // NOLINT(readability-static-accessed-through-instance)
    while (clock_current - clock_begin < million) {
      clock_current = clock64();
    }
  }

  *out = static_cast<int64_t>(clock_current);
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

static void run_prewarm(rmm::cuda_stream_pool& stream_pool, rmm::device_async_resource_ref mr)
{
  auto buffers = std::vector<rmm::device_uvector<int64_t>>();
  for (int32_t i = 0; i < stream_pool.get_pool_size(); i++) {
    auto stream = stream_pool.get_stream(i);
    buffers.emplace_back(rmm::device_uvector<int64_t>(1, stream, mr));
  }
}

static void run_test(std::size_t num_kernels,
                     rmm::cuda_stream_pool& stream_pool,
                     rmm::device_async_resource_ref mr)
{
  for (int32_t i = 0; i < num_kernels; i++) {
    auto stream = stream_pool.get_stream(i);
    auto buffer = rmm::device_uvector<int64_t>(1, stream, mr);
    compute_bound_kernel<<<1, 1, 0, stream.value()>>>(buffer.data());
  }
}

static void BM_MultiStreamAllocations(benchmark::State& state, MRFactoryFunc const& factory)
{
  auto mr = factory();

  rmm::mr::set_current_device_resource_ref(mr.get());

  auto num_streams = state.range(0);
  auto num_kernels = state.range(1);
  bool do_prewarm  = state.range(2) != 0;

  auto stream_pool = rmm::cuda_stream_pool(num_streams);

  if (do_prewarm) { run_prewarm(stream_pool, mr.get()); }

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    run_test(num_kernels, stream_pool, mr.get());
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_kernels));

  rmm::mr::reset_current_device_resource_ref();
}

inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_cuda_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda(), rmm::percent_of_free_device_memory(50));
}

inline auto make_arena()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
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

static void benchmark_range(benchmark::internal::Benchmark* bench)
{
  bench  //
    ->RangeMultiplier(2)
    ->Ranges({{1, 4}, {4, 4}, {false, true}})
    ->Unit(benchmark::kMicrosecond);
}

MRFactoryFunc get_mr_factory(std::string const& resource_name)
{
  if (resource_name == "cuda") { return &make_cuda; }
  if (resource_name == "cuda_async") { return &make_cuda_async; }
  if (resource_name == "pool") { return &make_pool; }
  if (resource_name == "arena") { return &make_arena; }
  if (resource_name == "binning") { return &make_binning; }

  RMM_FAIL("Invalid memory_resource name: " + resource_name);
}

void declare_benchmark(std::string const& name)
{
  if (name == "cuda") {
    BENCHMARK_CAPTURE(BM_MultiStreamAllocations, cuda, &make_cuda)  //
      ->Apply(benchmark_range);
    return;
  }

  if (name == "cuda_async") {
    BENCHMARK_CAPTURE(BM_MultiStreamAllocations, cuda_async, &make_cuda_async)  //
      ->Apply(benchmark_range);
    return;
  }

  if (name == "pool") {
    BENCHMARK_CAPTURE(BM_MultiStreamAllocations, pool_mr, &make_pool)  //
      ->Apply(benchmark_range);
    return;
  }

  if (name == "arena") {
    BENCHMARK_CAPTURE(BM_MultiStreamAllocations, arena, &make_arena)  //
      ->Apply(benchmark_range);
    return;
  }

  if (name == "binning") {
    BENCHMARK_CAPTURE(BM_MultiStreamAllocations, binning, &make_binning)  //
      ->Apply(benchmark_range);
    return;
  }

  RMM_FAIL("Invalid memory_resource name: " + name);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void run_profile(std::string const& resource_name, int kernel_count, int stream_count, bool prewarm)
{
  auto mr_factory  = get_mr_factory(resource_name);
  auto mr          = mr_factory();
  auto stream_pool = rmm::cuda_stream_pool(stream_count);

  if (prewarm) { run_prewarm(stream_pool, mr.get()); }

  run_test(kernel_count, stream_pool, mr.get());
}

int main(int argc, char** argv)
{
  try {
    ::benchmark::Initialize(&argc, argv);

    // Parse for replay arguments:
    cxxopts::Options options(
      "RMM Multi Stream Allocations Benchmark",
      "Benchmarks interleaving temporary allocations with compute-bound kernels.");

    options.add_options()(  //
      "p,profile",
      "Profiling mode: run once",
      cxxopts::value<bool>()->default_value("false"));

    options.add_options()(  //
      "r,resource",
      "Type of device_memory_resource",
      cxxopts::value<std::string>()->default_value("pool"));

    options.add_options()(  //
      "k,kernels",
      "Number of kernels to run: (default: 8)",
      cxxopts::value<int>()->default_value("8"));

    options.add_options()(  //
      "s,streams",
      "Number of streams in stream pool (default: 8)",
      cxxopts::value<int>()->default_value("8"));

    options.add_options()(  //
      "w,warm",
      "Ensure each stream has enough memory to satisfy allocations.",
      cxxopts::value<bool>()->default_value("false"));

    auto args = options.parse(argc, argv);

    if (args.count("profile") > 0) {
      auto resource_name = args["resource"].as<std::string>();
      auto num_kernels   = args["kernels"].as<int>();
      auto num_streams   = args["streams"].as<int>();
      auto prewarm       = args["warm"].as<bool>();
      try {
        run_profile(resource_name, num_kernels, num_streams, prewarm);
      } catch (std::exception const& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
      }
    } else {
      auto resource_names = std::vector<std::string>();

      if (args.count("resource") > 0) {
        resource_names.emplace_back(args["resource"].as<std::string>());
      } else {
        resource_names.emplace_back("cuda");
        resource_names.emplace_back("cuda_async");
        resource_names.emplace_back("pool");
        resource_names.emplace_back("arena");
        resource_names.emplace_back("binning");
      }

      for (auto& resource_name : resource_names) {
        declare_benchmark(resource_name);
      }

      ::benchmark::RunSpecifiedBenchmarks();
    }
  } catch (std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
