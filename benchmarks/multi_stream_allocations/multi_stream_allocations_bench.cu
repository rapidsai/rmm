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

#include <cuda_runtime_api.h>

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

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

static void BM_MultiStreamAllocations(benchmark::State& state, MRFactoryFunc factory)
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

inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_cuda_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
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
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool, 18, 22);
  return mr;
}

static void benchmark_range(benchmark::internal::Benchmark* b)
{
  b  //
    ->RangeMultiplier(2)
    ->Ranges({{1, 4}, {4, 4}, {false, true}})
    ->Unit(benchmark::kMicrosecond);
}

void declare_benchmark()
{
  BENCHMARK_CAPTURE(BM_MultiStreamAllocations, cuda, &make_cuda)  //
    ->Apply(benchmark_range);
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  BENCHMARK_CAPTURE(BM_MultiStreamAllocations, cuda_async, &make_cuda_async)  //
    ->Apply(benchmark_range);
#endif
  BENCHMARK_CAPTURE(BM_MultiStreamAllocations, pool_mr, &make_pool)  //
    ->Apply(benchmark_range);

  BENCHMARK_CAPTURE(BM_MultiStreamAllocations, arena, &make_arena)  //
    ->Apply(benchmark_range);

  BENCHMARK_CAPTURE(BM_MultiStreamAllocations, binning, &make_binning)  //
    ->Apply(benchmark_range);
}

int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  declare_benchmark();
  ::benchmark::RunSpecifiedBenchmarks();
}
