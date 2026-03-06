/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/thread_safe_resource_adaptor.hpp>

namespace rmm::test {

struct ThreadSafeMRFixture : public ::testing::Test {
  rmm::mr::thread_safe_resource_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(ThreadSafeMR, CcclMrRefTest, ThreadSafeMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(ThreadSafeMR, CcclMrRefAllocationTest, ThreadSafeMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(ThreadSafeMR, CcclMrRefTestMT, ThreadSafeMRFixture);

}  // namespace rmm::test
