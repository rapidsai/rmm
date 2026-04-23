/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/process_is_exiting.hpp>

#include <gtest/gtest.h>

/*
 * Tests for rmm::process_is_exiting() and the internal register_process_exit_hook().
 *
 * These tests cover the observable behavior of the API during normal program execution. The
 * complementary property -- that the flag flips to true before the destructor of a static
 * object whose construction was sequenced-before register_process_exit_hook() -- is exercised
 * by the standalone PROCESS_IS_EXITING_SHUTDOWN_TEST binary, whose exit code is checked by
 * CTest.
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

// register_process_exit_hook is idempotent: repeated calls are safe and do not flip the flag
// before the actual exit.
TEST(ProcessIsExitingTest, RegisterHookIsIdempotent)
{
  rmm::detail::register_process_exit_hook();
  rmm::detail::register_process_exit_hook();
  rmm::detail::register_process_exit_hook();
  EXPECT_FALSE(rmm::process_is_exiting());
}
