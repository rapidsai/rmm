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
  rmm::mr::set_current_device_resource(rmm::mr::cuda_memory_resource{});
  auto old = rmm::mr::set_current_device_resource(this->ref);

  // Old ref should be functional (verify by successful allocation)
  constexpr std::size_t size{100};
  void* ptr =
    old.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  old.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // Current device resource should be usable for allocation
  auto current = rmm::mr::get_current_device_resource_ref();
  ptr =
    current.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  test_get_current_device_resource_ref();

  // Resetting should reset to initial cuda resource
  rmm::mr::reset_current_device_resource();
  // Verify reset worked by checking allocation succeeds with initial resource
  current = rmm::mr::get_current_device_resource_ref();
  ptr =
    current.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  current.deallocate(
    cuda::stream_ref{cudaStream_t{nullptr}}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST_P(mr_ref_test, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TEST_P(mr_ref_test, AllocationsAreDifferent) { concurrent_allocations_are_different(this->ref); }

TEST_P(mr_ref_test, AsyncAllocationsAreDifferentDefaultStream)
{
  concurrent_async_allocations_are_different(this->ref, cuda::stream_ref{cudaStream_t{nullptr}});
}

TEST_P(mr_ref_test, AsyncAllocationsAreDifferent)
{
  concurrent_async_allocations_are_different(this->ref, this->stream);
}

}  // namespace rmm::test
