/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

namespace rmm::test {

struct BinningMRFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::binning_memory_resource mr{upstream, 18, 22};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(BinningMR, CcclMrRefTest, BinningMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(BinningMR, CcclMrRefAllocationTest, BinningMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(BinningMR, CcclMrRefTestMT, BinningMRFixture);

}  // namespace rmm::test
