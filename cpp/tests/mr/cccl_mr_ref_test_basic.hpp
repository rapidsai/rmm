/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test.hpp"

namespace rmm::test {

/**
 * @brief Typed-test fixture for basic CCCL-style memory resource tests.
 *
 * The Fixture parameter must be a ::testing::Test subclass providing:
 *   rmm::device_async_resource_ref ref
 *   rmm::cuda_stream stream
 */
template <typename Fixture>
struct CcclMrRefTest : public Fixture {};

TYPED_TEST_SUITE_P(CcclMrRefTest);

TYPED_TEST_P(CcclMrRefTest, SetCurrentDeviceResourceRef)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  auto cuda_ref = rmm::device_async_resource_ref{cuda_mr};

  rmm::mr::set_current_device_resource_ref(cuda_ref);
  auto old = rmm::mr::set_current_device_resource_ref(this->ref);

  constexpr std::size_t size{100};
  void* ptr = old.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  old.deallocate(rmm::cuda_stream_default, ptr, size);

  auto current = rmm::mr::get_current_device_resource_ref();
  ptr          = current.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(rmm::cuda_stream_default, ptr, size);

  test_get_current_device_resource_ref();

  rmm::mr::reset_current_device_resource_ref();
  current = rmm::mr::get_current_device_resource_ref();
  ptr     = current.allocate(rmm::cuda_stream_default, size);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(rmm::cuda_stream_default, ptr, size);
}

TYPED_TEST_P(CcclMrRefTest, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TYPED_TEST_P(CcclMrRefTest, AllocationsAreDifferent)
{
  concurrent_allocations_are_different(this->ref);
}

TYPED_TEST_P(CcclMrRefTest, AsyncAllocationsAreDifferentDefaultStream)
{
  concurrent_async_allocations_are_different(this->ref, cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefTest, AsyncAllocationsAreDifferent)
{
  concurrent_async_allocations_are_different(this->ref, this->stream);
}

REGISTER_TYPED_TEST_SUITE_P(CcclMrRefTest,
                            SetCurrentDeviceResourceRef,
                            SelfEquality,
                            AllocationsAreDifferent,
                            AsyncAllocationsAreDifferentDefaultStream,
                            AsyncAllocationsAreDifferent);

}  // namespace rmm::test
