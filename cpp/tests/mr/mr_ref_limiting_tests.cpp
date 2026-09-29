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
  static constexpr std::size_t allocation_limit{1_GiB};
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::limiting_resource_adaptor mr{upstream, allocation_limit};
  cuda::mr::resource_ref<cuda::mr::device_accessible> ref{mr};
  rmm::cuda_stream stream{};
};

// Multithreaded random allocation tests must not exceed the limit, even in the worst case.
static_assert(default_num_threads * default_num_allocations * default_max_size <=
              LimitingMRFixture::allocation_limit);

INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefTest, LimitingMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefAllocationTest, LimitingMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(LimitingMR, CcclMrRefTestMT, LimitingMRFixture);

}  // namespace rmm::test
