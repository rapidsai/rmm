/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>

namespace rmm::test {

struct FailureCallbackMRFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::failure_callback_resource_adaptor<> mr{
    upstream, [](std::size_t, void*) { return false; }, nullptr};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(FailureCallbackMR, CcclMrRefTest, FailureCallbackMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(FailureCallbackMR,
                               CcclMrRefAllocationTest,
                               FailureCallbackMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(FailureCallbackMR, CcclMrRefTestMT, FailureCallbackMRFixture);

}  // namespace rmm::test
