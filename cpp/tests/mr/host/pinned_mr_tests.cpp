/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Suppress deprecation warnings for testing deprecated functionality
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

// Returns true if a pointer points to pinned host memory.
inline bool is_pinned_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return attributes.type == cudaMemoryTypeHost;
}

}  // namespace

// Issue 2057: Ensure the inherited host_memory_resource overload allocate(bytes) is usable
TEST(PinnedMemoryResource, AllocateBytesOverload)
{
  rmm::mr::pinned_memory_resource mr;

  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(128));
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_pinned_memory(ptr));
  EXPECT_NO_THROW(mr.deallocate_sync(ptr, 128));
}

}  // namespace rmm::test

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
