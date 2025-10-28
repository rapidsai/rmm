/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/cxxopts.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace {

/**
 * @brief Factory function to create a cuda_async_memory_resource with priming
 */
inline auto make_cuda_async_primed()
{
  auto const [free, total] = rmm::available_device_memory();
  auto const pool_size     = total / 2;
  return std::make_shared<rmm::mr::cuda_async_memory_resource>(pool_size);
}

/**
 * @brief Factory function to create a cuda_async_memory_resource without priming
 */
inline auto make_cuda_async_unprimed()
{
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

/**
 * @brief Benchmark to measure the impact of async allocator priming
 */
template <typename MRFactoryFunc>
void BM_AsyncPrimingImpact(benchmark::State& state, MRFactoryFunc factory)
{
  auto const [free, total]   = rmm::available_device_memory();
  auto const allocation_size = static_cast<std::size_t>(total * 0.009);  // 0.9% of total memory
  auto const num_allocations = 100;

  // Create memory resource
  auto mr = factory();

  // Storage for allocations
  std::vector<void*> allocations;
  allocations.reserve(num_allocations);

  for (auto _ : state) {
    // Measure latency to first allocation
    auto start_time = std::chrono::high_resolution_clock::now();

    // First allocation - measure latency to this specific call
    allocations.push_back(mr->allocate_sync(allocation_size));
    cudaDeviceSynchronize();
    auto first_allocation_time = std::chrono::high_resolution_clock::now();

    // Continue with remaining allocations in first round
    for (int i = 1; i < num_allocations; ++i) {
      allocations.push_back(mr->allocate_sync(allocation_size));
    }

    cudaDeviceSynchronize();
    auto first_round_end = std::chrono::high_resolution_clock::now();

    // Deallocate all
    for (auto* ptr : allocations) {
      mr->deallocate_sync(ptr, allocation_size);
    }
    allocations.clear();

    // Second round of allocations
    for (int i = 0; i < num_allocations; ++i) {
      allocations.push_back(mr->allocate_sync(allocation_size));
    }

    cudaDeviceSynchronize();
    auto second_round_end = std::chrono::high_resolution_clock::now();

    // Calculate metrics
    auto latency_to_first =
      std::chrono::duration_cast<std::chrono::nanoseconds>(first_allocation_time - start_time)
        .count();
    auto first_round_duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(first_round_end - start_time).count();
    auto second_round_duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(second_round_end - first_round_end)
        .count();

    // Calculate throughput (bytes per second)
    auto first_round_throughput =
      (static_cast<double>(num_allocations * allocation_size) * 1e9) / first_round_duration_ns;
    auto second_round_throughput =
      (static_cast<double>(num_allocations * allocation_size) * 1e9) / second_round_duration_ns;

    // Set benchmark counters
    state.counters["latency_to_first_ns"]     = latency_to_first;
    state.counters["first_round_throughput"]  = first_round_throughput;
    state.counters["second_round_throughput"] = second_round_throughput;

    // Clean up for next iteration
    for (auto* ptr : allocations) {
      mr->deallocate_sync(ptr, allocation_size);
    }
    allocations.clear();
  }
}

/**
 * @brief Benchmark to measure construction time with and without priming
 */
template <typename MRFactoryFunc>
void BM_AsyncConstructionTime(benchmark::State& state, MRFactoryFunc factory)
{
  for (auto _ : state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mr         = factory();
    auto end_time   = std::chrono::high_resolution_clock::now();

    auto construction_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    state.counters["construction_time_ns"] = construction_time;
  }
}

}  // namespace

// Register benchmarks
BENCHMARK_CAPTURE(BM_AsyncPrimingImpact, primed, &make_cuda_async_primed)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_AsyncPrimingImpact, unprimed, &make_cuda_async_unprimed)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, primed, &make_cuda_async_primed)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, unprimed, &make_cuda_async_unprimed)
  ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv)
{
  try {
    ::benchmark::Initialize(&argc, argv);

    cxxopts::Options options("async_priming_bench", "Benchmark async allocator priming impact");
    options.add_options()("h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

  } catch (std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
