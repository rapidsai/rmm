/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/cxxopts.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

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

using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
using MRFactoryFunc       = std::function<any_device_resource()>;

namespace {

constexpr std::size_t cross_stream_operations = 1024;
constexpr std::size_t cross_stream_pool_bytes = cross_stream_operations * 65536;

std::pair<any_device_resource, cudaMemPool_t> make_cross_stream_pool(bool use_cccl)
{
  if (!use_cccl) {
    auto pool = rmm::mr::cuda_async_memory_resource{};
    return {pool, pool.pool_handle()};
  }

  // Match RMM's native creation flags. CCCL's memory_pool_properties does not expose
  // the hardware-decompression flag, which RMM enables on supported CUDA versions.
  cudaMemPoolProps properties{};
  properties.allocType     = cudaMemAllocationTypePinned;
  properties.handleTypes   = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id   = rmm::get_current_cuda_device().value();
#if CUDART_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  if (rmm::detail::hwdecompress::is_supported()) {
    properties.usage = cudaMemPoolCreateUsageHwDecompress;
  }
#endif
  cudaMemPool_t handle{};
  RMM_CUDA_TRY(cudaMemPoolCreate(&handle, &properties));
  return {cuda::shared_device_memory_pool::from_native_handle(handle), handle};
}

void BM_CrossStreamOwnership(benchmark::State& state, bool use_cccl)
try {
  auto [pool, handle] = make_cross_stream_pool(use_cccl);
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto const bytes = static_cast<std::size_t>(state.range(0));

  // Scratch does not use the measured pool. Each allocation touches only its first
  // 64-bit word; allocation size is varied independently of copy size.
  rmm::mr::cuda_memory_resource scratch;
  rmm::device_uvector<std::uint64_t> inputs{cross_stream_operations, producer, scratch};
  rmm::device_uvector<std::uint64_t> outputs{cross_stream_operations, consumer, scratch};
  std::vector<std::uint64_t> expected(cross_stream_operations), actual(cross_stream_operations);
  auto const result_bytes = cross_stream_operations * sizeof(std::uint64_t);
  cuda::event ready{cuda::device_ref{rmm::get_current_cuda_device().value()}};

  try {
    // Use identical retention/reuse policies and explicitly finish priming before
    // either pool is used. Even a whole batch of outstanding allocations fits.
    std::uint64_t threshold = cross_stream_pool_bytes;
    RMM_CUDA_TRY(cudaMemPoolSetAttribute(handle, cudaMemPoolAttrReleaseThreshold, &threshold));
    for (auto attribute : {cudaMemPoolReuseFollowEventDependencies,
                           cudaMemPoolReuseAllowOpportunistic,
                           cudaMemPoolReuseAllowInternalDependencies}) {
      int enabled = 1;
      RMM_CUDA_TRY(cudaMemPoolSetAttribute(handle, attribute, &enabled));
    }
    {
      rmm::device_buffer prime{cross_stream_pool_bytes, producer, resource};
    }
    producer.synchronize();

    std::uint64_t epoch = 0;
    auto run_batch      = [&] {
      // Inputs stay immutable until all consumers finish. Distinct nonzero values
      // detect stale results across both operations and batches, including warmup.
      std::iota(expected.begin(), expected.end(), ++epoch * cross_stream_operations);
      RMM_CUDA_TRY(cudaMemcpyAsync(
        inputs.data(), expected.data(), result_bytes, cudaMemcpyHostToDevice, producer.value()));
      RMM_CUDA_TRY(cudaMemsetAsync(outputs.data(), 0, result_bytes, consumer.value()));
      producer.synchronize();
      consumer.synchronize();

      using clock      = std::chrono::steady_clock;
      auto const start = clock::now();
      for (std::size_t i = 0; i < cross_stream_operations; ++i) {
        rmm::device_buffer buffer{bytes, producer, resource};
        RMM_CUDA_TRY(cudaMemcpyAsync(buffer.data(),
                                     inputs.data() + i,
                                     sizeof(std::uint64_t),
                                     cudaMemcpyDeviceToDevice,
                                     producer.value()));
        RMM_CUDA_TRY(cudaEventRecord(ready.get(), producer.value()));
        RMM_CUDA_TRY(cudaStreamWaitEvent(consumer.value(), ready.get(), 0));
        // The wait captures this record; later records do not change an enqueued wait.
        buffer.set_stream(consumer);
        RMM_CUDA_TRY(cudaMemcpyAsync(outputs.data() + i,
                                     buffer.data(),
                                     sizeof(std::uint64_t),
                                     cudaMemcpyDeviceToDevice,
                                     consumer.value()));
      }  // Destruction enqueues each free on the consumer, after its last use.
      auto const submitted = clock::now();
      // The final event wait covers all producer work. Finish all frees before
      // reusing scratch or destroying the pool; no implicit sync is assumed.
      consumer.synchronize();
      auto const completed = clock::now();

      RMM_CUDA_TRY(cudaMemcpyAsync(
        actual.data(), outputs.data(), result_bytes, cudaMemcpyDeviceToHost, consumer.value()));
      consumer.synchronize();
      RMM_EXPECTS(actual == expected, "Cross-stream data validation failed");
      return std::pair{std::chrono::duration<double>(submitted - start).count(),
                       std::chrono::duration<double>(completed - start).count()};
    };

    run_batch();  // Untimed, fully validated warmup.
    double submit_seconds = 0;
    for (auto _ : state) {
      auto const [submit, completion] = run_batch();
      state.SetIterationTime(completion);
      submit_seconds += submit;
    }
    // Both intervals are host wall time per batch. Manual real_time and throughput
    // use completion time. Setup, priming, input/reset and host validation are excluded.
    state.counters["host_submit_wall_us"] =
      benchmark::Counter(submit_seconds * 1e6, benchmark::Counter::kAvgIterations);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(cross_stream_operations));
  } catch (...) {
    // Keep the pool and scratch alive while draining any partially submitted batch.
    producer.synchronize_no_throw();
    consumer.synchronize_no_throw();
    throw;
  }
} catch (std::exception const& error) {
  // Report setup failures as well as submission/validation failures on this case.
  state.SkipWithError(error.what());
}

void declare_cross_stream_benchmarks()
{
  for (auto const& [name, use_cccl] : {std::pair{"cuda_async", false}, {"cuda_shared", true}}) {
    benchmark::RegisterBenchmark(
      (std::string{"BM_CrossStreamOwnership/"} + name).c_str(), BM_CrossStreamOwnership, use_cccl)
      ->Arg(4096)
      ->Arg(65536)
      ->ArgName("bytes")
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  }
}

}  // namespace

static void run_prewarm(rmm::cuda_stream_pool& stream_pool, rmm::device_async_resource_ref mr)
{
  auto buffers = std::vector<rmm::device_uvector<int64_t>>();
  for (std::size_t i = 0; i < stream_pool.get_pool_size(); i++) {
    auto stream = stream_pool.get_stream(i);
    buffers.emplace_back(rmm::device_uvector<int64_t>(1, stream, mr));
  }
}

static void run_test(std::size_t num_kernels,
                     rmm::cuda_stream_pool& stream_pool,
                     rmm::device_async_resource_ref mr)
{
  for (std::size_t i = 0; i < num_kernels; i++) {
    auto stream = stream_pool.get_stream(i);
    auto buffer = rmm::device_uvector<int64_t>(1, stream, mr);
    compute_bound_kernel<<<1, 1, 0, stream.get()>>>(buffer.data());
  }
}

static void BM_MultiStreamAllocations(benchmark::State& state, MRFactoryFunc const& factory)
{
  auto mr = factory();

  auto num_streams = state.range(0);
  auto num_kernels = state.range(1);
  bool do_prewarm  = state.range(2) != 0;

  auto stream_pool = rmm::cuda_stream_pool(static_cast<std::size_t>(num_streams));

  if (do_prewarm) { run_prewarm(stream_pool, mr); }

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    run_test(static_cast<std::size_t>(num_kernels), stream_pool, mr);
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_kernels));
}

inline any_device_resource make_cuda() { return rmm::mr::cuda_memory_resource{}; }

inline any_device_resource make_cuda_async() { return rmm::mr::cuda_async_memory_resource{}; }

inline any_device_resource make_pool()
{
  return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                       rmm::percent_of_free_device_memory(50)};
}

inline any_device_resource make_arena()
{
  return rmm::mr::arena_memory_resource{rmm::mr::get_current_device_resource_ref()};
}

inline any_device_resource make_binning()
{
  // Add a binning_memory_resource with fixed-size bins of sizes 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  constexpr auto min_bin_pow2{18};
  constexpr auto max_bin_pow2{22};
  auto pool = make_pool();
  return rmm::mr::binning_memory_resource{pool, min_bin_pow2, max_bin_pow2};
}

static void benchmark_range(benchmark::Benchmark* bench)
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
  auto stream_pool = rmm::cuda_stream_pool(static_cast<std::size_t>(stream_count));

  if (prewarm) { run_prewarm(stream_pool, mr); }

  run_test(static_cast<std::size_t>(kernel_count), stream_pool, mr);
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
      "Type of memory resource",
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
        declare_cross_stream_benchmarks();
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
