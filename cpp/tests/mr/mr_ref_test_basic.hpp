/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

  // old mr should equal a cuda mr
  EXPECT_EQ(old, cuda_ref);

  // current dev resource should equal this resource
  EXPECT_EQ(this->ref, rmm::mr::get_current_device_resource_ref());

  test_get_current_device_resource_ref();

  // Resetting should reset to initial cuda resource
  rmm::mr::reset_current_device_resource_ref();
  EXPECT_EQ(rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()},
            rmm::mr::get_current_device_resource_ref());
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
