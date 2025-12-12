/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test_allocation.hpp"
#include "mr_ref_test_basic.hpp"

namespace rmm::test {
namespace {

INSTANTIATE_TEST_SUITE_P(SystemResourceTests,
                         mr_ref_test,
                         ::testing::Values("System"),
                         [](auto const& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(SystemResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("System"),
                         [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
