/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"
#include "mr_ref_test_allocation.hpp"
#include "mr_ref_test_basic.hpp"
#include "mr_ref_test_mt.hpp"

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace rmm::test {
namespace {

INSTANTIATE_TEST_SUITE_P(ArenaResourceTests,
                         mr_ref_test,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(ArenaResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(ArenaMultiThreadResourceTests,
                         mr_ref_test_mt,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

}  // namespace

struct ArenaMRFixture : public ::testing::Test {
  rmm::mr::arena_memory_resource mr{rmm::mr::get_current_device_resource_ref()};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(ArenaMR, CcclMrRefTest, ArenaMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(ArenaMR, CcclMrRefAllocationTest, ArenaMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(ArenaMR, CcclMrRefTestMT, ArenaMRFixture);

}  // namespace rmm::test
