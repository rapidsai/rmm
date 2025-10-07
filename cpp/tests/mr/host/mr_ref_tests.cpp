/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "../../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <random>

namespace rmm::test {
namespace {
inline bool is_aligned(void* ptr, std::size_t alignment = alignof(std::max_align_t))
{
  return rmm::is_pointer_aligned(ptr, alignment);
}

// Returns true if a pointer points to a device memory or managed memory allocation.
inline bool is_device_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return (attributes.type == cudaMemoryTypeDevice) or (attributes.type == cudaMemoryTypeManaged);
}

/**
 * @brief Returns if a pointer `p` points to pinned host memory.
 */
inline bool is_pinned_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return attributes.type == cudaMemoryTypeHost;
}

constexpr std::size_t size_word{4_B};
constexpr std::size_t size_kb{1_KiB};
constexpr std::size_t size_mb{1_MiB};
constexpr std::size_t size_gb{1_GiB};
constexpr std::size_t size_pb{1_PiB};

struct allocation {
  void* ptr{nullptr};
  std::size_t size{0};
  allocation(void* ptr, std::size_t size) : ptr{ptr}, size{size} {}
  allocation() = default;
};
}  // namespace

template <typename MemoryResourceType>
struct MRRefTest : public ::testing::Test {
  MemoryResourceType mr;
  rmm::host_resource_ref ref;

  MRRefTest() : mr{}, ref{mr} {}
};

using resources = ::testing::Types<rmm::mr::new_delete_resource, rmm::mr::pinned_memory_resource>;

// static property checks
static_assert(
  rmm::detail::polyfill::resource_with<rmm::mr::new_delete_resource, cuda::mr::host_accessible>);
static_assert(
  rmm::detail::polyfill::resource_with<rmm::mr::pinned_memory_resource, cuda::mr::host_accessible>);

TYPED_TEST_SUITE(MRRefTest, resources);

TYPED_TEST(MRRefTest, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

TYPED_TEST(MRRefTest, AllocateZeroBytes)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->ref.allocate(0));
  EXPECT_NO_THROW(this->ref.deallocate(ptr, 0));
}

TYPED_TEST(MRRefTest, AllocateWord)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->ref.allocate(size_word));
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_aligned(ptr));
  EXPECT_FALSE(is_device_memory(ptr));
  EXPECT_NO_THROW(this->ref.deallocate(ptr, size_word));
}

TYPED_TEST(MRRefTest, AllocateKB)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->ref.allocate(size_kb));
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_aligned(ptr));
  EXPECT_FALSE(is_device_memory(ptr));
  EXPECT_NO_THROW(this->ref.deallocate(ptr, size_kb));
}

TYPED_TEST(MRRefTest, AllocateMB)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->ref.allocate(size_mb));
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_aligned(ptr));
  EXPECT_FALSE(is_device_memory(ptr));
  EXPECT_NO_THROW(this->ref.deallocate(ptr, size_mb));
}

TYPED_TEST(MRRefTest, AllocateGB)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->ref.allocate(size_gb));
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_aligned(ptr));
  EXPECT_FALSE(is_device_memory(ptr));
  EXPECT_NO_THROW(this->ref.deallocate(ptr, size_gb));
}

TYPED_TEST(MRRefTest, AllocateTooMuch)
{
  void* ptr{nullptr};
  EXPECT_THROW(ptr = this->ref.allocate(size_pb), std::bad_alloc);
  EXPECT_EQ(nullptr, ptr);
}

TYPED_TEST(MRRefTest, RandomAllocations)
{
  constexpr std::size_t num_allocations{100};
  std::vector<allocation> allocations(num_allocations);

  constexpr std::size_t MAX_ALLOCATION_SIZE{5 * size_mb};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, MAX_ALLOCATION_SIZE);

  // 100 allocations from [0,5MB)
  std::for_each(
    allocations.begin(), allocations.end(), [&generator, &distribution, this](allocation& alloc) {
      alloc.size = distribution(generator);
      EXPECT_NO_THROW(alloc.ptr = this->ref.allocate(alloc.size));
      EXPECT_NE(nullptr, alloc.ptr);
      EXPECT_TRUE(is_aligned(alloc.ptr));
    });

  std::for_each(allocations.begin(), allocations.end(), [this](allocation& alloc) {
    EXPECT_NO_THROW(this->ref.deallocate(alloc.ptr, alloc.size));
  });
}

TYPED_TEST(MRRefTest, MixedRandomAllocationFree)
{
  std::default_random_engine generator;

  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(1, MAX_ALLOCATION_SIZE);

  // How often a free will occur. For example, if `1`, then every allocation
  // will immediately be free'd. Or, if 4, on average, a free will occur after
  // every 4th allocation
  constexpr std::size_t FREE_FREQUENCY{4};
  std::uniform_int_distribution<int> free_distribution(1, FREE_FREQUENCY);

  std::deque<allocation> allocations;

  constexpr std::size_t num_allocations{100};
  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t allocation_size = size_distribution(generator);
    EXPECT_NO_THROW(allocations.emplace_back(this->ref.allocate(allocation_size), allocation_size));
    auto new_allocation = allocations.back();
    EXPECT_NE(nullptr, new_allocation.ptr);
    EXPECT_TRUE(is_aligned(new_allocation.ptr));

    bool const free_front{free_distribution(generator) == free_distribution.max()};

    if (free_front) {
      auto front = allocations.front();
      EXPECT_NO_THROW(this->ref.deallocate(front.ptr, front.size));
      allocations.pop_front();
    }
  }
  // free any remaining allocations
  for (auto alloc : allocations) {
    EXPECT_NO_THROW(this->ref.deallocate(alloc.ptr, alloc.size));
    allocations.pop_front();
  }
}

static constexpr std::size_t MinTestedAlignment{16};
static constexpr std::size_t MaxTestedAlignment{4096};
static constexpr std::size_t TestedAlignmentMultiplier{2};
static constexpr std::size_t NUM_TRIALS{100};

TYPED_TEST(MRRefTest, AlignmentTest)
{
  std::default_random_engine generator(0);
  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(1, MAX_ALLOCATION_SIZE);

  for (std::size_t num_trials = 0; num_trials < NUM_TRIALS; ++num_trials) {
    for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
         alignment *= TestedAlignmentMultiplier) {
      auto allocation_size = size_distribution(generator);
      void* ptr{nullptr};
      EXPECT_NO_THROW(ptr = this->ref.allocate(allocation_size, alignment));
      EXPECT_TRUE(is_aligned(ptr, alignment));
      EXPECT_NO_THROW(this->ref.deallocate(ptr, allocation_size, alignment));
    }
  }
}

TYPED_TEST(MRRefTest, UnsupportedAlignmentTest)
{
  std::default_random_engine generator(0);
  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(1, MAX_ALLOCATION_SIZE);

  for (std::size_t num_trials = 0; num_trials < NUM_TRIALS; ++num_trials) {
    for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
         alignment *= TestedAlignmentMultiplier) {
#ifdef NDEBUG
      auto allocation_size = size_distribution(generator);
      void* ptr{nullptr};
      // An unsupported alignment (like an odd number) should result in an
      // alignment of `alignof(std::max_align_t)`
      auto const bad_alignment = alignment + 1;

      EXPECT_NO_THROW(ptr = this->ref.allocate(allocation_size, bad_alignment));
      EXPECT_TRUE(is_aligned(ptr, alignof(std::max_align_t)));
      EXPECT_NO_THROW(this->ref.deallocate(ptr, allocation_size, bad_alignment));
#endif
    }
  }
}

TEST(PinnedResource, isPinned)
{
  rmm::mr::pinned_memory_resource mr;
  rmm::host_resource_ref ref{mr};
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = ref.allocate(100));
  EXPECT_TRUE(is_pinned_memory(ptr));
  EXPECT_NO_THROW(ref.deallocate(ptr, 100));
}
}  // namespace rmm::test
