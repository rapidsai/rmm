/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::binning_memory_resource<rmm::mr::cuda_memory_resource>;

namespace rmm::test {

using cuda_mr    = rmm::mr::cuda_memory_resource;
using binning_mr = rmm::mr::binning_memory_resource<cuda_mr>;

TEST(BinningTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { binning_mr mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(BinningTest, ExplicitBinMR)
{
  cuda_mr cuda{};
  binning_mr mr{&cuda};
  mr.add_bin(1024, &cuda);
  auto* ptr = mr.allocate_sync(512);
  EXPECT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, 512);
}

}  // namespace rmm::test
