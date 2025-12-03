/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

// Suppress warnings about uninstantiated tests (Fixed_Size only has basic tests)
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(mr_ref_allocation_test);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(mr_ref_test_mt);

// Single-threaded basic tests (5 tests)
// Note: Fixed_Size MR cannot handle dynamic allocation sizes, so only basic tests are included
INSTANTIATE_TEST_SUITE_P(FixedSizeResourceTests,
                         mr_ref_test,
                         ::testing::Values("Fixed_Size"),
                         [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
