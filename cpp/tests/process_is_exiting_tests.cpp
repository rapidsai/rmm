/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/process_is_exiting.hpp>

#include <gtest/gtest.h>

/*
 * Tests for rmm::process_is_exiting().
 *
 * These tests cover the observable behavior of the API during normal program execution. The
 * complementary property -- that the flag flips to true before the destructor of the
 * per-device resource map returned by detail::get_ref_map() -- is exercised by the standalone
 * PROCESS_IS_EXITING_SHUTDOWN_TEST binary, whose exit code is checked by CTest.
 */

// Flag should be false while the test process is running normally.
TEST(ProcessIsExitingTest, FalseDuringNormalExecution) { EXPECT_FALSE(rmm::process_is_exiting()); }

// Calling the query is safe to call repeatedly and does not modify state.
TEST(ProcessIsExitingTest, QueryIsIdempotent)
{
  EXPECT_FALSE(rmm::process_is_exiting());
  EXPECT_FALSE(rmm::process_is_exiting());
  EXPECT_FALSE(rmm::process_is_exiting());
}
