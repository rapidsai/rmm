/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../mock_resource.hpp"

#include <rmm/error.hpp>
#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_mr    = rmm::mr::cuda_memory_resource;
using binning_mr = rmm::mr::binning_memory_resource;

TEST(BinningTest, ExplicitBinMR)
{
  cuda_mr cuda{};
  binning_mr mr{cuda};
  mr.add_bin(1024, rmm::device_async_resource_ref{cuda});
  auto* ptr = mr.allocate_sync(512);
  EXPECT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, 512);
}

TEST(BinningTest, ZeroByteAllocationsUseBinResource)
{
  cuda_mr cuda{};
  mock_resource mock;
  mock_resource_wrapper wrapper{&mock};
  binning_mr mr{cuda};
  mr.add_bin(1024, device_async_resource_ref{wrapper});

  EXPECT_CALL(mock, allocate(::testing::_, 0, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .Times(2)
    .WillRepeatedly(::testing::Return(nullptr));
  EXPECT_CALL(mock, deallocate(::testing::_, nullptr, 0, rmm::CUDA_ALLOCATION_ALIGNMENT)).Times(2);

  EXPECT_EQ(mr.allocate(cuda_stream_view{}, 0, rmm::CUDA_ALLOCATION_ALIGNMENT), nullptr);
  mr.deallocate(cuda_stream_view{}, nullptr, 0, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(mr.allocate_sync(0, rmm::CUDA_ALLOCATION_ALIGNMENT), nullptr);
  mr.deallocate_sync(nullptr, 0, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

}  // namespace

}  // namespace rmm::test
