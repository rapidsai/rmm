/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

// Suppress deprecation warnings for testing deprecated functionality
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;

namespace rmm::test {
namespace {
using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;

TEST(PinnedPoolTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { pool_mr mr{nullptr, 1024}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(PinnedPoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{&pinned_mr, initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PinnedPoolTest, ReferenceThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{pinned_mr, initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

// Issue #527
TEST(PinnedPoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    pool_mr mr(pinned_mr, 1000192, 1000192);
    mr.allocate_sync(1000);
  }());
}

TEST(PinnedPoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      rmm::mr::pinned_memory_resource pinned_mr{};
      pool_mr mr(pinned_mr, 1000031, 1000192);
      mr.allocate_sync(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      rmm::mr::pinned_memory_resource pinned_mr{};
      pool_mr mr(pinned_mr, 1000192, 1000200);
      mr.allocate_sync(1000);
    }(),
    rmm::logic_error);
}

TEST(PinnedPoolTest, ThrowOutOfMemory)
{
  rmm::mr::pinned_memory_resource pinned_mr{};
  const auto initial{0};
  const auto maximum{1024};
  pool_mr mr{pinned_mr, initial, maximum};
  mr.allocate_sync(1024);

  EXPECT_THROW(mr.allocate_sync(1024), rmm::out_of_memory);
}

}  // namespace
}  // namespace rmm::test

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
