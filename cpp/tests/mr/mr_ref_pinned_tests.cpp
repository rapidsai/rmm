/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

// Suppress warning about uninstantiated multi-threaded tests (Pinned doesn't support MT tests)
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(mr_ref_test_mt);

// Single-threaded basic tests (5 tests)
INSTANTIATE_TEST_SUITE_P(PinnedResourceTests,
                         mr_ref_test,
                         ::testing::Values("Pinned"),
                         [](auto const& info) { return info.param; });

// Single-threaded allocation tests (9 tests)
INSTANTIATE_TEST_SUITE_P(PinnedResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("Pinned"),
                         [](auto const& info) { return info.param; });

// Note: No multi-threaded tests for Pinned memory resource

}  // namespace
}  // namespace rmm::test
