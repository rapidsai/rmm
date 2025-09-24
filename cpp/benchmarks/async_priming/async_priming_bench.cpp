/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
  std::cout << "[DEBUG] Creating primed async MR - Free: " << (free / (1024 * 1024)) << " MB, "
            << "Total: " << (total / (1024 * 1024)) << " MB, "
            << "Pool size: " << (pool_size / (1024 * 1024)) << " MB" << std::endl;
  return std::make_shared<rmm::mr::cuda_async_memory_resource>(pool_size);
}

/**
 * @brief Factory function to create a cuda_async_memory_resource without priming
 */
inline auto make_cuda_async_unprimed()
{
  auto const [free, total] = rmm::available_device_memory();
  std::cout << "[DEBUG] Creating unprimed async MR - Free: " << (free / (1024 * 1024)) << " MB, "
            << "Total: " << (total / (1024 * 1024)) << " MB" << std::endl;
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

/**
 * @brief Benchmark to measure the impact of async allocator priming
 */
template <typename MRFactoryFunc>
void BM_AsyncPrimingImpact(benchmark::State& state, MRFactoryFunc factory)
{
  auto const [free, total]   = rmm::available_device_memory();
  auto const allocation_size = static_cast<std::size_t>(total * 0.09);  // 9% of total memory
  auto const num_allocations = 10;

  std::cout << "[DEBUG] Starting benchmark - Allocation size: " << (allocation_size / (1024 * 1024))
            << " MB, Num allocations: " << num_allocations << std::endl;

  // Create memory resource
  auto mr = factory();

  // Storage for allocations
  std::vector<void*> allocations;
  allocations.reserve(num_allocations);

  for (auto _ : state) {
    // Measure latency to first allocation
    auto start_time = std::chrono::high_resolution_clock::now();

    // First allocation - measure latency to this specific call
    std::cout << "[DEBUG] Starting first allocation..." << std::endl;
    allocations.push_back(mr->allocate(allocation_size));
    auto first_allocation_time = std::chrono::high_resolution_clock::now();
    auto first_latency =
      std::chrono::duration_cast<std::chrono::microseconds>(first_allocation_time - start_time)
        .count();
    std::cout << "[DEBUG] First allocation completed in " << first_latency << " μs" << std::endl;

    // Continue with remaining allocations in first round
    std::cout << "[DEBUG] Continuing with remaining " << (num_allocations - 1) << " allocations..."
              << std::endl;
    for (int i = 1; i < num_allocations; ++i) {
      allocations.push_back(mr->allocate(allocation_size));
    }

    auto first_round_end = std::chrono::high_resolution_clock::now();
    auto first_round_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(first_round_end - start_time).count();
    std::cout << "[DEBUG] First round completed in " << first_round_duration << " μs" << std::endl;

    // Deallocate all
    std::cout << "[DEBUG] Deallocating all " << allocations.size() << " allocations..."
              << std::endl;
    for (auto* ptr : allocations) {
      mr->deallocate(ptr, allocation_size);
    }
    allocations.clear();

    // Second round of allocations
    std::cout << "[DEBUG] Starting second round of allocations..." << std::endl;
    for (int i = 0; i < num_allocations; ++i) {
      allocations.push_back(mr->allocate(allocation_size));
    }

    auto second_round_end = std::chrono::high_resolution_clock::now();
    auto second_round_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(second_round_end - first_round_end)
        .count();
    std::cout << "[DEBUG] Second round completed in " << second_round_duration << " μs"
              << std::endl;

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

    // Debug output for metrics
    std::cout << "[DEBUG] Metrics - Latency to first: " << (latency_to_first / 1000.0) << " μs, "
              << "First round throughput: " << (first_round_throughput / 1e9) << " GB/s, "
              << "Second round throughput: " << (second_round_throughput / 1e9) << " GB/s"
              << std::endl;

    // Set benchmark counters
    state.counters["latency_to_first_ns"]          = latency_to_first;
    state.counters["first_round_throughput_GBps"]  = first_round_throughput / 1e9;
    state.counters["second_round_throughput_GBps"] = second_round_throughput / 1e9;
    state.counters["allocation_size_MB"]           = allocation_size / (1024 * 1024);
    state.counters["num_allocations"]              = num_allocations;

    // Clean up for next iteration
    for (auto* ptr : allocations) {
      mr->deallocate(ptr, allocation_size);
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
    std::cout << "[DEBUG] Starting construction benchmark..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mr         = factory();
    auto end_time   = std::chrono::high_resolution_clock::now();

    auto construction_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    std::cout << "[DEBUG] Construction completed in " << (construction_time / 1000.0) << " μs"
              << std::endl;

    state.counters["construction_time_ns"] = construction_time;
  }
}

}  // namespace

// Register benchmarks
BENCHMARK_CAPTURE(BM_AsyncPrimingImpact, primed, &make_cuda_async_primed)
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

BENCHMARK_CAPTURE(BM_AsyncPrimingImpact, unprimed, &make_cuda_async_unprimed)
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, primed, &make_cuda_async_primed)
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, unprimed, &make_cuda_async_unprimed)
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

int main(int argc, char** argv)
{
  try {
    std::cout << "[DEBUG] Starting async priming benchmark..." << std::endl;

    cxxopts::Options options("async_priming_bench", "Benchmark async allocator priming impact");
    options.add_options()("h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

  } catch (std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
