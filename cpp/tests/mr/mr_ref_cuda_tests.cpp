/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test_allocation.hpp"
#include "mr_ref_test_basic.hpp"
#include "mr_ref_test_mt.hpp"

namespace rmm::test {
namespace {

INSTANTIATE_TEST_SUITE_P(CudaResourceTests,
                         mr_ref_test,
                         ::testing::Values("CUDA"),
                         [](auto const& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(CudaResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("CUDA"),
                         [](auto const& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(CudaMultiThreadResourceTests,
                         mr_ref_test_mt,
                         ::testing::Values("CUDA"),
                         [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
