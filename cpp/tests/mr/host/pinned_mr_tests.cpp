/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

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
