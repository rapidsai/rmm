/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <memory_resource>
#include <random>
#include <vector>

namespace rmm::test {

namespace {
template <typename MemoryResourceType>
class UnsynchronizedPoolTest : public ::testing::Test {
 protected:
  MemoryResourceType upstream_mr;
  std::pmr::unsynchronized_pool_resource pool_mr{&upstream_mr};
};
}  // namespace

using resources = ::testing::Types<rmm::mr::new_delete_resource, rmm::mr::pinned_memory_resource>;
TYPED_TEST_SUITE(UnsynchronizedPoolTest, resources);

TYPED_TEST(UnsynchronizedPoolTest, ZeroBytesAllocation)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->pool_mr.allocate(0));
  EXPECT_NO_THROW(this->pool_mr.deallocate(ptr, 0));
}

TYPED_TEST(UnsynchronizedPoolTest, MultipleAllocations)
{
  constexpr std::size_t num_allocations{100};
  std::vector<void*> ptrs;
  std::vector<std::size_t> sizes;

  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t size = (i + 1) * 1024;  // Different sizes for each allocation
    EXPECT_NE(nullptr, ptrs.emplace_back(this->pool_mr.allocate(size)));
    sizes.push_back(size);
  }

  // Deallocate in reverse order
  for (std::size_t i = num_allocations - 1; i > 0; --i) {
    EXPECT_NO_THROW(this->pool_mr.deallocate(ptrs[i], sizes[i]));
  }
}

TYPED_TEST(UnsynchronizedPoolTest, RandomAllocations)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> size_distribution(1, 1024 * 1024);  // 1B to 1MB

  constexpr std::size_t num_allocations{100};
  std::vector<void*> ptrs;
  std::vector<std::size_t> sizes;

  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t size = size_distribution(generator);
    EXPECT_NE(nullptr, ptrs.emplace_back(this->pool_mr.allocate(size)));
    sizes.push_back(size);
  }

  // Deallocate in random order
  std::vector<std::size_t> indices(num_allocations);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), generator);

  for (std::size_t idx : indices) {
    EXPECT_NO_THROW(this->pool_mr.deallocate(ptrs[idx], sizes[idx]));
  }
}

TYPED_TEST(UnsynchronizedPoolTest, AlignmentTest)
{
  constexpr std::size_t sizes[]      = {16, 32, 64, 128, 256, 512, 1024};
  constexpr std::size_t alignments[] = {8, 16, 32, 64, 128, 256, 512};

  for (std::size_t size : sizes) {
    for (std::size_t alignment : alignments) {
      void* ptr = this->pool_mr.allocate(size, alignment);
      EXPECT_NE(nullptr, ptr);
      EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(ptr) % alignment);
      this->pool_mr.deallocate(ptr, size, alignment);
    }
  }
}

// Special test for pinned memory resource to verify memory is actually pinned
TEST(PinnedUnsynchronizedPoolTest, VerifyPinnedMemory)
{
  rmm::mr::pinned_memory_resource pinned_mr;
  std::pmr::unsynchronized_pool_resource pool_mr{&pinned_mr};

  void* ptr = pool_mr.allocate(1024);
  EXPECT_NE(nullptr, ptr);

  cudaPointerAttributes attributes{};
  EXPECT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_EQ(cudaMemoryTypeHost, attributes.type);

  pool_mr.deallocate(ptr, 1024);
}

}  // namespace rmm::test