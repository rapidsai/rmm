/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace rmm::test {

// Note: Fixed_Size MR cannot handle dynamic allocation sizes, so only basic tests are included
struct FixedSizeMRFixture : public ::testing::Test {
  rmm::mr::fixed_size_memory_resource mr{rmm::mr::get_current_device_resource_ref()};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(FixedSizeMR, CcclMrRefTest, FixedSizeMRFixture);

}  // namespace rmm::test
