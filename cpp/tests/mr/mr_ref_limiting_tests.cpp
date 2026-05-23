/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>

namespace rmm::test {

struct LimitingMRFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::limiting_resource_adaptor mr{upstream, 1ULL << 30};  // 1 GiB limit
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefTest, LimitingMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefAllocationTest, LimitingMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefTestMT, LimitingMRFixture);

}  // namespace rmm::test
