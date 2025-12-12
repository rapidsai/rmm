/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test_basic.hpp"

namespace rmm::test {
namespace {

// Note: Fixed_Size MR cannot handle dynamic allocation sizes, so only basic tests are included
INSTANTIATE_TEST_SUITE_P(FixedSizeResourceTests,
                         mr_ref_test,
                         ::testing::Values("Fixed_Size"),
                         [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
