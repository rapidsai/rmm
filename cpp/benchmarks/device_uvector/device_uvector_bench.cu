/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "../synchronization/synchronization.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <benchmark/benchmark.h>

#include <cstdio>
#include <type_traits>

void BM_UvectorSizeConstruction(benchmark::State& state)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    &cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource_ref(mr);

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    rmm::device_uvector<std::int32_t> vec(state.range(0), rmm::cuda_stream_view{});
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));

  rmm::mr::reset_current_device_resource_ref();
}

BENCHMARK(BM_UvectorSizeConstruction)
  ->RangeMultiplier(10)           // NOLINT
  ->Range(10'000, 1'000'000'000)  // NOLINT
  ->Unit(benchmark::kMicrosecond);

void BM_ThrustVectorSizeConstruction(benchmark::State& state)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
    &cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource_ref(mr);

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    rmm::device_vector<std::int32_t> vec(state.range(0));
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));

  rmm::mr::reset_current_device_resource_ref();
}

BENCHMARK(BM_ThrustVectorSizeConstruction)
  ->RangeMultiplier(10)           // NOLINT
  ->Range(10'000, 1'000'000'000)  // NOLINT
  ->Unit(benchmark::kMicrosecond);

// simple kernel used to test concurrent execution.
__global__ void kernel(int const* input, int* output, std::size_t num)
{
  for (auto i = blockDim.x * blockIdx.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    output[i] = input[i] * input[i];
  }
}

using thrust_vector = thrust::device_vector<int32_t>;
using rmm_vector    = rmm::device_vector<int32_t>;
using rmm_uvector   = rmm::device_uvector<int32_t>;

template <typename Vector>
Vector make_vector(std::int64_t num_elements, rmm::cuda_stream_view stream, bool zero_init = false)
{
  static_assert(std::is_same_v<Vector, thrust_vector> or std::is_same_v<Vector, rmm_vector> or
                  std::is_same_v<Vector, rmm_uvector>,
                "unsupported vector type");
  if constexpr (std::is_same_v<Vector, thrust_vector>) {
    return Vector(num_elements, 0);
  } else if constexpr (std::is_same_v<Vector, rmm_vector>) {
    return Vector(num_elements, 0, rmm::mr::thrust_allocator<std::int32_t>(stream));
  } else if constexpr (std::is_same_v<Vector, rmm_uvector>) {
    auto vec = Vector(num_elements, stream);
    if (zero_init) {
      cudaMemsetAsync(vec.data(), 0, num_elements * sizeof(std::int32_t), stream.value());
    }
    return vec;
  }
}

template <typename Vector>
int32_t* vector_data(Vector& vec)
{
  return thrust::raw_pointer_cast(vec.data());
}

template <typename Vector>
void vector_workflow(std::size_t num_elements,
                     std::int64_t num_blocks,
                     std::int64_t block_size,
                     rmm::cuda_stream const& input_stream,
                     std::vector<rmm::cuda_stream> const& streams)
{
  auto input = make_vector<Vector>(num_elements, input_stream, true);
  input_stream.synchronize();
  for (rmm::cuda_stream_view stream : streams) {
    auto output = make_vector<Vector>(num_elements, stream);
    kernel<<<num_blocks, block_size, 0, stream.value()>>>(
      vector_data(input), vector_data(output), num_elements);
  }

  for (rmm::cuda_stream_view stream : streams) {
    stream.synchronize();
  }
}

template <typename Vector>
void BM_VectorWorkflow(benchmark::State& state)
{
  rmm::mr::cuda_async_memory_resource cuda_async_mr{};
  rmm::mr::set_current_device_resource_ref(cuda_async_mr);

  rmm::cuda_stream input_stream;
  std::vector<rmm::cuda_stream> streams(4);

  auto const num_elements   = state.range(0);
  auto constexpr block_size = 256;
  auto constexpr num_blocks = 16;

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    cuda_event_timer timer(state, true, input_stream);  // flush_l2_cache = true
    vector_workflow<Vector>(num_elements, num_blocks, block_size, input_stream, streams);
  }

  auto constexpr num_accesses = 9;
  auto const bytes            = num_elements * sizeof(std::int32_t) * num_accesses;
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));

  rmm::mr::reset_current_device_resource_ref();
}

BENCHMARK_TEMPLATE(BM_VectorWorkflow, thrust_vector)  // NOLINT
  ->RangeMultiplier(10)                               // NOLINT
  ->Range(100'000, 100'000'000)                       // NOLINT
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

// The only difference here is that `rmm::device_vector` uses
// `rmm::get_current_device_resource_ref()` for allocation while `thrust::device_vector` uses
// cudaMalloc/cudaFree. In the benchmarks we use `cuda_async_memory_resource`, which is faster.
BENCHMARK_TEMPLATE(BM_VectorWorkflow, rmm_vector)  // NOLINT
  ->RangeMultiplier(10)                            // NOLINT
  ->Range(100'000, 100'000'000)                    // NOLINT
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_VectorWorkflow, rmm_uvector)  // NOLINT
  ->RangeMultiplier(10)                             // NOLINT
  ->Range(100'000, 100'000'000)                     // NOLINT
  ->Unit(benchmark::kMicrosecond)
  ->UseManualTime();

BENCHMARK_MAIN();
