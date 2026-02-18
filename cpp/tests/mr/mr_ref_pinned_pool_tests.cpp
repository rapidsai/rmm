/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace rmm::test {

struct PinnedPoolMRFixture : public ::testing::Test {
  rmm::mr::pinned_host_memory_resource pinned{};
  rmm::mr::pool_memory_resource mr{pinned, 0};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(PinnedPoolMR, CcclMrRefTest, PinnedPoolMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(PinnedPoolMR, CcclMrRefAllocationTest, PinnedPoolMRFixture);

}  // namespace rmm::test
