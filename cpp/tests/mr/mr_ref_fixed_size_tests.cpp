/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>

namespace rmm::test {

// Note: Fixed_Size MR cannot handle dynamic allocation sizes, so only basic tests are included
struct FixedSizeMRFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::fixed_size_memory_resource mr{upstream};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(FixedSizeMR, CcclMrRefTest, FixedSizeMRFixture);

}  // namespace rmm::test
