/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>

static void BM_UvectorSizeConstruction(benchmark::State& state)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&mr);

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    rmm::device_uvector<int32_t> vec(state.range(0), rmm::cuda_stream_view{});
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

  rmm::mr::set_current_device_resource(nullptr);
}

BENCHMARK(BM_UvectorSizeConstruction)
  ->RangeMultiplier(10)           // NOLINT
  ->Range(10'000, 1'000'000'000)  // NOLINT
  ->Unit(benchmark::kMicrosecond);

static void BM_ThrustVectorSizeConstruction(benchmark::State& state)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&mr);

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    rmm::device_vector<int32_t> vec(state.range(0));
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

  rmm::mr::set_current_device_resource(nullptr);
}

BENCHMARK(BM_ThrustVectorSizeConstruction)
  ->RangeMultiplier(10)           // NOLINT
  ->Range(10'000, 1'000'000'000)  // NOLINT
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
