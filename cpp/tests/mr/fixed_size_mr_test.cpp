/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/cmath>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>

#include <gtest/gtest.h>

#include <cstddef>
#include <future>
#include <mutex>
#include <string>
#include <vector>

using namespace rmm::test;

namespace {

struct FixedSizeMRTestParam {
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream;
  std::string name;           ///< Name of the memory resource.
  std::size_t block_size{0};  ///< Block size in bytes.
  std::size_t size{0};        ///< Allocation size in bytes.
  std::size_t n_threads{1};   ///< Number of threads to use for the test.
  std::size_t n_streams{1};   ///< Number of streams to use for the test.
};

std::vector<FixedSizeMRTestParam> make_fixed_size_mr_test_params()
{
  std::vector<FixedSizeMRTestParam> params;

  auto add_params = [&](cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                        std::string const& name) {
    for (std::size_t block_sz : {std::size_t{256}, std::size_t{1_KiB}}) {
      for (std::size_t size : {std::size_t{0}, block_sz / 2, block_sz, std::size_t{3_KiB}}) {
        for (std::size_t n_threads : std::vector<std::size_t>{1, 2, 4}) {
          for (std::size_t n_streams : std::vector<std::size_t>{1, 2, 4}) {
            params.emplace_back(upstream, name, block_sz, size, n_threads, n_streams);
          }
        }
      }
    }
  };

  add_params(rmm::mr::cuda_memory_resource{}, "Cuda");
  add_params(rmm::mr::cuda_async_memory_resource{}, "CudaAsync");

  return params;
}

}  // namespace

class FixedSizeMRTest : public ::testing::TestWithParam<FixedSizeMRTestParam> {
 protected:
  using stats_mr          = rmm::mr::statistics_resource_adaptor;
  using fixed_size_mr     = rmm::mr::fixed_size_memory_resource;
  using multi_block_alloc = rmm::mr::multiple_blocks_allocation;

  std::size_t expected_blocks()
  {
    auto const& param = GetParam();
    if (param.size == 0) { return std::size_t{0}; }
    return cuda::ceil_div(param.size, param.block_size);
  }
};

TEST_P(FixedSizeMRTest, AllocAndDeallocBlocksAsync)
{
  auto const& param                           = GetParam();
  constexpr std::size_t blocks_to_preallocate = 1;

  // statistics_resource_adaptor is itself a shared_resource: copying it shares the same pool.
  auto counting = stats_mr(param.upstream);

  {
    auto fixed_mr = fixed_size_mr(counting, param.block_size, blocks_to_preallocate);

    rmm::cuda_stream_pool stream_pool{param.n_streams, rmm::cuda_stream::flags::non_blocking};

    std::size_t const alloc_size = param.size;
    std::size_t const n_threads  = param.n_threads;
    std::size_t const block_size = fixed_mr.get_block_size();
    EXPECT_EQ(block_size, param.block_size);

    constexpr int num_allocations = 16;
    std::mutex handles_mutex;
    std::vector<std::unique_ptr<multi_block_alloc>> handles;

    std::vector<std::future<void>> alloc_futs;
    alloc_futs.reserve(param.n_threads);

    // each thread allocates num_allocations allocations
    for (std::size_t i = 0; i < n_threads; ++i) {
      alloc_futs.emplace_back(std::async(std::launch::async, [&] {
        for (int i = 0; i < num_allocations; ++i) {
          rmm::cuda_stream_view stream = stream_pool.get_stream();
          auto handle = multi_block_alloc::make_async(fixed_mr, alloc_size, stream);
          EXPECT_EQ(handle->size(), alloc_size);
          EXPECT_EQ(handle->capacity(), expected_blocks() * block_size);

          // enqueue a dummy cudamemsetasync
          int dummy = 0;
          cuda::std::ranges::for_each(handle->get_blocks(), [&](auto& block) {
            RMM_CUDA_TRY(cudaMemsetAsync(block, (dummy++) & 0xFF, block_size, stream.value()));
          });

          {
            std::lock_guard lock(handles_mutex);
            handles.emplace_back(std::move(handle));
          }
        }
      }));
    }

    // wait for all allocations to complete
    cuda::std::ranges::for_each(alloc_futs, [](auto& fut) { fut.get(); });

    // Note that stream pool is not sync'ed. The counter & driver should be able to account for the
    // allocations without sync.
    auto const bytes_after_alloc = counting.get_bytes_counter().value;

    if (expected_blocks() > 0) {
      EXPECT_GE(
        bytes_after_alloc,
        static_cast<std::int64_t>(expected_blocks() * num_allocations * block_size * n_threads));
    }

    // deallocate using multiple threads
    std::vector<std::future<void>> dealloc_futs;
    dealloc_futs.reserve(param.n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      dealloc_futs.emplace_back(std::async(std::launch::async, [&] {
        while (true) {
          std::lock_guard lock(handles_mutex);
          if (handles.empty()) { break; }
          handles.pop_back();
        }
      }));
    }

    // wait for all deallocations to complete
    cuda::std::ranges::for_each(dealloc_futs, [](auto& fut) { fut.get(); });

    EXPECT_EQ(counting.get_bytes_counter().value, bytes_after_alloc)
      << "After deallocate, upstream bytes must be unchanged until fixed_size_mr is destroyed";

    // finally sync the stream pool
    for (size_t i = 0; i < param.n_streams; ++i) {
      stream_pool.get_stream(i).synchronize();
    }
  }

  EXPECT_EQ(counting.get_bytes_counter().value, 0)
    << "After fixed_size pool destruction, upstream should have released all memory";
}

INSTANTIATE_TEST_SUITE_P(FixedSizeMRTests,
                         FixedSizeMRTest,
                         ::testing::ValuesIn(make_fixed_size_mr_test_params()),
                         [](testing::TestParamInfo<FixedSizeMRTestParam> const& info) {
                           return info.param.name + "_bs" + std::to_string(info.param.block_size) +
                                  "_sz" + std::to_string(info.param.size) + "_nt" +
                                  std::to_string(info.param.n_threads) + "_ns" +
                                  std::to_string(info.param.n_streams);
                         });
