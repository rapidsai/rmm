/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace rmm::test {

struct PoolMRFixture : public ::testing::Test {
  rmm::mr::pool_memory_resource mr{rmm::mr::get_current_device_resource_ref(), 0};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(PoolMR, CcclMrRefTest, PoolMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(PoolMR, CcclMrRefAllocationTest, PoolMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(PoolMR, CcclMrRefTestMT, PoolMRFixture);

}  // namespace rmm::test
