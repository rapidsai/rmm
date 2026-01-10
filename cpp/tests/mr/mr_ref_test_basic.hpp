/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test.hpp"

namespace rmm::test {

// Parameterized test definitions for mr_ref_test (basic tests)

TEST_P(mr_ref_test, SetCurrentDeviceResourceRef)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  auto cuda_ref = rmm::device_async_resource_ref{cuda_mr};

  rmm::mr::set_current_device_resource_ref(cuda_ref);
  auto old = rmm::mr::set_current_device_resource_ref(this->ref);

  // Old ref should be functional (verify by successful allocation)
  constexpr std::size_t size{100};
  void* ptr = old.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  old.deallocate(rmm::cuda_stream_default, ptr, size);

  // Current device resource should be usable for allocation
  auto current = rmm::mr::get_current_device_resource_ref();
  ptr          = current.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(rmm::cuda_stream_default, ptr, size);

  test_get_current_device_resource_ref();

  // Resetting should reset to initial cuda resource
  rmm::mr::reset_current_device_resource_ref();
  // Verify reset worked by checking allocation succeeds with initial resource
  current = rmm::mr::get_current_device_resource_ref();
  ptr     = current.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(rmm::cuda_stream_default, ptr, size);
}

TEST_P(mr_ref_test, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TEST_P(mr_ref_test, AllocationsAreDifferent) { concurrent_allocations_are_different(this->ref); }

TEST_P(mr_ref_test, AsyncAllocationsAreDifferentDefaultStream)
{
  concurrent_async_allocations_are_different(this->ref, cuda_stream_view{});
}

TEST_P(mr_ref_test, AsyncAllocationsAreDifferent)
{
  concurrent_async_allocations_are_different(this->ref, this->stream);
}

}  // namespace rmm::test
