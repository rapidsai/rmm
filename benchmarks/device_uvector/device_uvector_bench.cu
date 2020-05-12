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

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>
#include "rmm/mr/device/cnmem_memory_resource.hpp"
#include "rmm/mr/device/default_memory_resource.hpp"

static void
BM_UvectorSizeConstruction(benchmark::State& state) {
  rmm::mr::cnmem_memory_resource mr{};
  rmm::mr::set_default_resource(&mr);

  for (auto _ : state) {
    rmm::device_uvector<int32_t>(state.range(0));
    cudaDeviceSynchronize();
  }
}
BENCHMARK(BM_UvectorSizeConstruction)
  ->RangeMultiplier(10)
  ->Range(10'000, 1'000'000'000)
  ->Unit(benchmark::kMicrosecond);

static void
BM_UvectorZeroInitializedConstruction(benchmark::State& state) {
  rmm::mr::cnmem_memory_resource mr{};
  rmm::mr::set_default_resource(&mr);

  for (auto _ : state) {
    rmm::device_uvector<int32_t>(state.range(0), int32_t{0});
    cudaDeviceSynchronize();
  }
}
BENCHMARK(BM_UvectorZeroInitializedConstruction)
  ->RangeMultiplier(10)
  ->Range(10'000, 1'000'000'000)
  ->Unit(benchmark::kMicrosecond);

static void
BM_UvectorInitializedConstruction(benchmark::State& state) {
  rmm::mr::cnmem_memory_resource mr{};
  rmm::mr::set_default_resource(&mr);

  for (auto _ : state) {
    rmm::device_uvector<int32_t>(state.range(0), int32_t{1});
    cudaDeviceSynchronize();
  }
}
BENCHMARK(BM_UvectorInitializedConstruction)
  ->RangeMultiplier(10)
  ->Range(10'000, 1'000'000'000)
  ->Unit(benchmark::kMicrosecond);

    static void BM_ThrustVectorSizeConstruction(benchmark::State& state) {
  rmm::mr::cnmem_memory_resource mr{};
  rmm::mr::set_default_resource(&mr);
  for (auto _ : state) {
    rmm::device_vector<int32_t>(state.range(0));
    cudaDeviceSynchronize();
  }
}

BENCHMARK(BM_ThrustVectorSizeConstruction)
  ->RangeMultiplier(10)
  ->Range(10'000, 1'000'000'000)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
