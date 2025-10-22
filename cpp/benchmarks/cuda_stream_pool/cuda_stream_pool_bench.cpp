/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <stdexcept>

static void BM_StreamPoolGetStream(benchmark::State& state)
{
  rmm::cuda_stream_pool stream_pool{};

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto stream = stream_pool.get_stream();
    cudaStreamQuery(stream.value());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_StreamPoolGetStream)->Unit(benchmark::kMicrosecond);

static void BM_CudaStreamClass(benchmark::State& state)
{
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto stream = rmm::cuda_stream{};
    cudaStreamQuery(stream.view().value());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_CudaStreamClass)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
