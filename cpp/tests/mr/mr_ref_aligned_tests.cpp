/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace rmm::test {

struct AlignedMRFixture : public ::testing::Test {
  rmm::mr::aligned_resource_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(AlignedMR, CcclMrRefTest, AlignedMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(AlignedMR, CcclMrRefAllocationTest, AlignedMRFixture);

}  // namespace rmm::test
