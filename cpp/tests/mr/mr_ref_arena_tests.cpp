/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test.hpp"

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

// Single-threaded basic tests (5 tests)
INSTANTIATE_TEST_SUITE_P(ArenaResourceTests,
                         mr_ref_test,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

// Single-threaded allocation tests (9 tests)
INSTANTIATE_TEST_SUITE_P(ArenaResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

// Multi-threaded tests (15 tests)
INSTANTIATE_TEST_SUITE_P(ArenaMultiThreadResourceTests,
                         mr_ref_test_mt,
                         ::testing::Values("Arena"),
                         [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
