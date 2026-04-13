/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {
using pool_mr = rmm::mr::pool_memory_resource;

TEST(PinnedPoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    pool_mr mr{rmm::mr::pinned_host_memory_resource{}, 1024, 256};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PinnedPoolTest, ReferenceThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    pool_mr mr{rmm::mr::pinned_host_memory_resource{}, 1024, 256};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

// Issue #527
TEST(PinnedPoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::pinned_host_memory_resource{}, 1000192, 1000192);
    (void)mr.allocate_sync(1000);
  }());
}

TEST(PinnedPoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::pinned_host_memory_resource{}, 1000031, 1000192);
      (void)mr.allocate_sync(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::pinned_host_memory_resource{}, 1000192, 1000200);
      (void)mr.allocate_sync(1000);
    }(),
    rmm::logic_error);
}

TEST(PinnedPoolTest, ThrowOutOfMemory)
{
  pool_mr mr{rmm::mr::pinned_host_memory_resource{}, 0, 1024};
  (void)mr.allocate_sync(1024);

  EXPECT_THROW((void)mr.allocate_sync(1024), rmm::out_of_memory);
}

}  // namespace
}  // namespace rmm::test
