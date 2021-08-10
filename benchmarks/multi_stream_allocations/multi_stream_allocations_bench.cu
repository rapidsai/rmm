/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

__global__ void compute_bound_kernel(int64_t* out)
{
  clock_t clock_begin   = clock64();
  clock_t clock_current = clock_begin;

  if (threadIdx.x == 0) {
    while (clock_current - clock_begin < 1000000) {
      clock_current = clock64();
    }
  }

  *out = static_cast<int64_t>(clock_current);
}

static void BM_MultiStreamAllocations(benchmark::State& state)
{
  auto cuda_mr = rmm::mr::cuda_memory_resource{};
  auto mr      = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>{&cuda_mr};

  rmm::mr::set_current_device_resource(&mr);

  auto num_streams = state.range(0);
  auto num_kernels = state.range(1);
  auto do_prewarm  = state.range(2);

  auto stream_pool = rmm::cuda_stream_pool(num_streams);

  if (do_prewarm) {
    auto buffers = std::vector<rmm::device_uvector<int64_t>>();
    for (int32_t i = 0; i < num_streams; i++) {
      auto stream = stream_pool.get_stream(i);
      buffers.emplace_back(rmm::device_uvector<int64_t>(1, stream, &mr));
    }
  }

  for (auto _ : state) {
    for (int32_t i = 0; i < num_kernels; i++) {
      auto stream = stream_pool.get_stream(i);
      auto buffer = rmm::device_uvector<int64_t>(1, stream, &mr);
      compute_bound_kernel<<<1, 1, 0, stream.value()>>>(buffer.data());
    }

    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(state.iterations() * num_kernels);

  rmm::mr::set_current_device_resource(nullptr);
}
BENCHMARK(BM_MultiStreamAllocations)
  ->RangeMultiplier(2)
  ->Ranges({{1, 16}, {8, 8}, {false, true}})
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
