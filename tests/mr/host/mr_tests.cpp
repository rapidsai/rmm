/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include <rmm/mr/host/host_memory_resource.hpp>
#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cstddef>
#include <deque>
#include <random>

namespace {
inline bool is_aligned(void* p, std::size_t alignment = alignof(std::max_align_t))
{
  return (0 == reinterpret_cast<uintptr_t>(p) % alignment);
}

inline void expect_aligned(void* p, std::size_t alignment)
{
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(p) % alignment);
}

/**---------------------------------------------------------------------------*
 * @brief Returns if a pointer points to a device memory or managed memory
 * allocation.
 *---------------------------------------------------------------------------**/
inline bool is_device_memory(void* p)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, p)) { return false; }
#if CUDART_VERSION < 10000  // memoryType is deprecated in CUDA 10
  return attributes.memoryType == cudaMemoryTypeDevice;
#else
  return (attributes.type == cudaMemoryTypeDevice) or (attributes.type == cudaMemoryTypeManaged);
#endif
}

/**
 * @brief Returns if a pointer `p` points to pinned host memory.
 */
inline bool is_pinned_memory(void* p)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, p)) { return false; }
  return attributes.type == cudaMemoryTypeHost;
}

static constexpr std::size_t size_word{4};
static constexpr std::size_t size_kb{std::size_t{1} << 10};
static constexpr std::size_t size_mb{std::size_t{1} << 20};
static constexpr std::size_t size_gb{std::size_t{1} << 30};
static constexpr std::size_t size_tb{std::size_t{1} << 40};
static constexpr std::size_t size_pb{std::size_t{1} << 50};

struct allocation {
  void* p{nullptr};
  std::size_t size{0};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};
}  // namespace

template <typename MemoryResourceType>
struct MRTest : public ::testing::Test {
  std::unique_ptr<rmm::mr::host_memory_resource> mr;

  MRTest() : mr{new MemoryResourceType} {}
  ~MRTest() = default;
};

using resources = ::testing::Types<rmm::mr::new_delete_resource, rmm::mr::pinned_memory_resource>;

TYPED_TEST_CASE(MRTest, resources);

TYPED_TEST(MRTest, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TYPED_TEST(MRTest, AllocateZeroBytes)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0));
  EXPECT_NO_THROW(this->mr->deallocate(p, 0));
}

TYPED_TEST(MRTest, AllocateWord)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_word));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_FALSE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_word));
}

TYPED_TEST(MRTest, AllocateKB)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_kb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_FALSE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_kb));
}

TYPED_TEST(MRTest, AllocateMB)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_mb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_FALSE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_mb));
}

TYPED_TEST(MRTest, AllocateGB)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_gb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_FALSE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_gb));
}

TYPED_TEST(MRTest, AllocateTooMuch)
{
  void* p{nullptr};
  EXPECT_THROW(p = this->mr->allocate(size_pb), std::bad_alloc);
  EXPECT_EQ(nullptr, p);
}

TYPED_TEST(MRTest, RandomAllocations)
{
  constexpr std::size_t num_allocations{100};
  std::vector<allocation> allocations(num_allocations);

  constexpr std::size_t MAX_ALLOCATION_SIZE{5 * size_mb};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, MAX_ALLOCATION_SIZE);

  // 100 allocations from [0,5MB)
  std::for_each(
    allocations.begin(), allocations.end(), [&generator, &distribution, this](allocation& a) {
      a.size = distribution(generator);
      EXPECT_NO_THROW(a.p = this->mr->allocate(a.size));
      EXPECT_NE(nullptr, a.p);
      EXPECT_TRUE(is_aligned(a.p));
    });

  std::for_each(
    allocations.begin(), allocations.end(), [generator, distribution, this](allocation& a) {
      EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size));
    });
}

TYPED_TEST(MRTest, MixedRandomAllocationFree)
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
    EXPECT_NO_THROW(allocations.emplace_back(this->mr->allocate(allocation_size), allocation_size));
    auto new_allocation = allocations.back();
    EXPECT_NE(nullptr, new_allocation.p);
    EXPECT_TRUE(is_aligned(new_allocation.p));

    bool const free_front{free_distribution(generator) == free_distribution.max()};

    if (free_front) {
      auto front = allocations.front();
      EXPECT_NO_THROW(this->mr->deallocate(front.p, front.size));
      allocations.pop_front();
    }
  }
  // free any remaining allocations
  for (auto a : allocations) {
    EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size));
    allocations.pop_front();
  }
}

static constexpr std::size_t MinTestedSize             = 32;
static constexpr std::size_t MaxTestedSize             = 8 * 1024;
static constexpr std::size_t TestedSizeStep            = 1;
static constexpr std::size_t MinTestedAlignment        = 16;
static constexpr std::size_t MaxTestedAlignment        = 4 * 1024;
static constexpr std::size_t TestedAlignmentMultiplier = 2;
static constexpr std::size_t NUM_TRIALS{100};

TYPED_TEST(MRTest, AlignmentTest)
{
  std::default_random_engine generator(0);
  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(1, MAX_ALLOCATION_SIZE);

  for (std::size_t num_trials = 0; num_trials < NUM_TRIALS; ++num_trials) {
    for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
         alignment *= TestedAlignmentMultiplier) {
      auto allocation_size = size_distribution(generator);
      void* ptr{nullptr};
      EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size, alignment));
      EXPECT_TRUE(is_aligned(ptr, alignment));
      EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size, alignment));
    }
  }
}

TYPED_TEST(MRTest, UnsupportedAlignmentTest)
{
  std::default_random_engine generator(0);
  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(1, MAX_ALLOCATION_SIZE);

  for (std::size_t num_trials = 0; num_trials < NUM_TRIALS; ++num_trials) {
    for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
         alignment *= TestedAlignmentMultiplier) {
      auto allocation_size = size_distribution(generator);
      void* ptr{nullptr};
      // An unsupported alignment (like an odd number) should result in an
      // alignment of `alignof(std::max_align_t)`
      auto const bad_alignment = alignment + 1;
      EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size, bad_alignment));
      expect_aligned(ptr, alignof(std::max_align_t));
      EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size, bad_alignment));
    }
  }
}

TEST(PinnedResource, isPinned)
{
  rmm::mr::pinned_memory_resource mr;
  void* p{nullptr};
  EXPECT_NO_THROW(p = mr.allocate(100));
  EXPECT_TRUE(is_pinned_memory(p));
  EXPECT_NO_THROW(mr.deallocate(p, 100));
}
