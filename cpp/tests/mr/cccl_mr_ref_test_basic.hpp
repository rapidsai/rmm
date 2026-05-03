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
  rmm::mr::set_current_device_resource(rmm::mr::cuda_memory_resource{});
  auto old = rmm::mr::set_current_device_resource(this->ref);

  constexpr std::size_t size{100};
  void* ptr =
    old.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  old.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto current = rmm::mr::get_current_device_resource_ref();
  ptr =
    current.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  test_get_current_device_resource_ref();

  rmm::mr::reset_current_device_resource();
  current = rmm::mr::get_current_device_resource_ref();
  ptr =
    current.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TYPED_TEST_P(CcclMrRefTest, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TYPED_TEST_P(CcclMrRefTest, AllocationsAreDifferent)
{
  concurrent_allocations_are_different(this->ref);
}

TYPED_TEST_P(CcclMrRefTest, AsyncAllocationsAreDifferentDefaultStream)
{
  concurrent_async_allocations_are_different(this->ref, cuda::stream_ref{cudaStream_t{nullptr}});
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
