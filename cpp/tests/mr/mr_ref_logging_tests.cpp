/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>

namespace rmm::test {

struct LoggingAdaptorFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource cuda{};
  rmm::mr::logging_resource_adaptor mr{cuda, "rmm_cccl_adaptor_test.txt"};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(LoggingAdaptor, CcclMrRefTest, LoggingAdaptorFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(LoggingAdaptor, CcclMrRefAllocationTest, LoggingAdaptorFixture);

}  // namespace rmm::test
