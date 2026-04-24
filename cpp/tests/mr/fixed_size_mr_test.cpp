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

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

namespace rmm::test {
namespace {

using stats_mr = rmm::mr::statistics_resource_adaptor;

struct FixedSizeMRTestParam {
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream;
  std::string name;
  std::size_t block_size{0};
  std::size_t size{0};
};

std::vector<FixedSizeMRTestParam> make_fixed_size_mr_test_params()
{
  std::vector<FixedSizeMRTestParam> params;

  auto add_params = [&](cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                        std::string const& name) {
    for (std::size_t block_sz : {std::size_t{256}, std::size_t{1_KiB}}) {
      for (std::size_t size : {std::size_t{0}, block_sz / 2, block_sz, std::size_t{3_KiB}}) {
        params.emplace_back(upstream, name, block_sz, size);
      }
    }
  };

  add_params(rmm::mr::cuda_memory_resource{}, "Cuda");
  add_params(rmm::mr::cuda_async_memory_resource{}, "CudaAsync");

  return params;
}

class FixedSizeMRTest : public ::testing::TestWithParam<FixedSizeMRTestParam> {};

TEST_P(FixedSizeMRTest, AllocateBlocksAsyncUpstreamCountedDeallocateDoesNotReturnToUpstream)
{
  auto const& param                           = GetParam();
  constexpr std::size_t blocks_to_preallocate = 1;

  // statistics_resource_adaptor is itself a shared_resource: copying it shares the same pool.
  auto counting = stats_mr(param.upstream);

  using fixed_size_mr = rmm::mr::fixed_size_memory_resource;

  {
    auto fixed_mr = fixed_size_mr(counting, param.block_size, blocks_to_preallocate);

    rmm::cuda_stream_pool stream_pool{4};
    std::vector<std::unique_ptr<rmm::mr::multiple_blocks_allocation>> handles;

    std::size_t const alloc_size  = param.size;
    constexpr int num_allocations = 4;

    std::size_t const actual_block_size = fixed_mr.get_block_size();

    std::size_t const expected_blocks = [&]() {
      if (alloc_size == 0) { return std::size_t{0}; }
      return cuda::ceil_div(alloc_size, actual_block_size);
    }();

    for (int i = 0; i < num_allocations; ++i) {
      rmm::cuda_stream_view stream = stream_pool.get_stream();
      auto const& handle           = handles.emplace_back(
        rmm::mr::multiple_blocks_allocation::make_async(fixed_mr, alloc_size, stream));

      EXPECT_EQ(handle->size(), alloc_size);
      EXPECT_EQ(handle->capacity(), expected_blocks * actual_block_size);
    }
    auto const bytes_after_alloc = counting.get_bytes_counter().value;

    if (expected_blocks > 0) {
      EXPECT_GE(bytes_after_alloc, static_cast<std::int64_t>(expected_blocks * num_allocations *
                                                              actual_block_size));
    }
    handles.clear();

    EXPECT_EQ(counting.get_bytes_counter().value, bytes_after_alloc)
      << "After deallocate, upstream bytes must be unchanged until fixed_size_mr is destroyed";
  }

  EXPECT_EQ(counting.get_bytes_counter().value, 0)
    << "After fixed_size pool destruction, upstream should have released all memory";
}

INSTANTIATE_TEST_SUITE_P(FixedSizeMRTests,
                         FixedSizeMRTest,
                         ::testing::ValuesIn(make_fixed_size_mr_test_params()),
                         [](testing::TestParamInfo<FixedSizeMRTestParam> const& info) {
                           return info.param.name + "_bs" +
                                  std::to_string(info.param.block_size) + "_sz" +
                                  std::to_string(info.param.size);
                         });

}  // namespace
}  // namespace rmm::test
