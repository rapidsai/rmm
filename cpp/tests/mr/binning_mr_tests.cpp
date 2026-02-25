/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>
#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {

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

}  // namespace rmm::test
