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

#include "gtest/gtest.h"

#include <rmm/mr/cnmem_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cstddef>
#include <deque>
#include <random>

namespace {
static constexpr std::size_t ALIGNMENT{256};
inline bool is_aligned(void* p, std::size_t alignment = ALIGNMENT) {
  return (0 == reinterpret_cast<uintptr_t>(p) % alignment);
}

/**---------------------------------------------------------------------------*
 * @brief Returns if a pointer points to a device memory or managed memory
 * allocation.
 *---------------------------------------------------------------------------**/
inline bool is_device_memory(void* p) {
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, p)) {
    return false;
  }
#if CUDART_VERSION < 10000  // memoryType is deprecated in CUDA 10
  return attributes.memoryType == cudaMemoryTypeDevice;
#else
  return (attributes.type == cudaMemoryTypeDevice) or
         (attributes.type == cudaMemoryTypeManaged);
#endif
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
  // some useful allocation sizes

  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  cudaStream_t stream;

  MRTest() : mr{new MemoryResourceType} {}

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };

  ~MRTest() = default;
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource
                                   rmm::mr::pinned_memory_resource>;

TYPED_TEST_CASE(MRTest, resources);

TEST(DefaultTest, UseDefaultResource) {
  EXPECT_NE(nullptr, rmm::mr::get_default_resource());
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_default_resource()->allocate(size_mb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_default_resource()->deallocate(p, size_mb));
}

TYPED_TEST(MRTest, SetDefaultResource) {
  EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
  rmm::mr::device_memory_resource* old{nullptr};
  EXPECT_NO_THROW(old = rmm::mr::set_default_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);
  EXPECT_TRUE(this->mr->is_equal(*rmm::mr::get_default_resource()));
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_default_resource()->allocate(size_mb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_default_resource()->deallocate(p, size_mb));
  // setting default resource w/ nullptr should reset to initial
  EXPECT_NO_THROW(rmm::mr::set_default_resource(nullptr));
  EXPECT_TRUE(old->is_equal(*rmm::mr::get_default_resource()));
  EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
}

TYPED_TEST(MRTest, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TYPED_TEST(MRTest, AllocateZeroBytes) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0));
  EXPECT_EQ(nullptr, p);
  EXPECT_NO_THROW(this->mr->deallocate(p, 0));
}

TYPED_TEST(MRTest, AllocateZeroBytesStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NO_THROW(this->mr->deallocate(p, 0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateWord) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_word));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_word));
}

TYPED_TEST(MRTest, AllocateWordStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_word, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_word, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateKB) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_kb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_kb));
}

TYPED_TEST(MRTest, AllocateKBStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_kb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_kb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateMB) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_mb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_mb));
}

TYPED_TEST(MRTest, AllocateMBStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_mb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_mb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateGB) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_gb));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_gb));
}

TYPED_TEST(MRTest, AllocateGBStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(size_gb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, size_gb, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateTooMuch) {
  void* p{nullptr};
  EXPECT_THROW(p = this->mr->allocate(size_pb), std::bad_alloc);
  EXPECT_EQ(nullptr, p);
}

TYPED_TEST(MRTest, AllocateTooMuchStream) {
  void* p{nullptr};
  EXPECT_THROW(p = this->mr->allocate(size_pb, this->stream), std::bad_alloc);
  EXPECT_EQ(nullptr, p);
}

TYPED_TEST(MRTest, RandomAllocations) {
  constexpr std::size_t num_allocations{100};
  std::vector<allocation> allocations(num_allocations);

  constexpr std::size_t MAX_ALLOCATION_SIZE{5 * size_mb};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1,
                                                          MAX_ALLOCATION_SIZE);

  // 100 allocations from [0,5MB)
  std::for_each(allocations.begin(), allocations.end(),
                [&generator, &distribution, this](allocation& a) {
                  a.size = distribution(generator);
                  EXPECT_NO_THROW(a.p = this->mr->allocate(a.size));
                  EXPECT_NE(nullptr, a.p);
                  EXPECT_TRUE(is_aligned(a.p));
                });

  std::for_each(allocations.begin(), allocations.end(),
                [generator, distribution, this](allocation& a) {
                  EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size));
                });
}

TYPED_TEST(MRTest, RandomAllocationsStream) {
  constexpr std::size_t num_allocations{100};
  std::vector<allocation> allocations(num_allocations);

  constexpr std::size_t MAX_ALLOCATION_SIZE{5 * size_mb};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1,
                                                          MAX_ALLOCATION_SIZE);

  // 100 allocations from [0,5MB)
  std::for_each(
      allocations.begin(), allocations.end(),
      [&generator, &distribution, this](allocation& a) {
        a.size = distribution(generator);
        EXPECT_NO_THROW(a.p = this->mr->allocate(a.size, this->stream));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
        EXPECT_NE(nullptr, a.p);
        EXPECT_TRUE(is_aligned(a.p));
      });

  std::for_each(
      allocations.begin(), allocations.end(),
      [generator, distribution, this](allocation& a) {
        EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size, this->stream));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
      });
}

TYPED_TEST(MRTest, MixedRandomAllocationFree) {
  std::default_random_engine generator;

  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(
      1, MAX_ALLOCATION_SIZE);

  // How often a free will occur. For example, if `1`, then every allocation
  // will immediately be free'd. Or, if 4, on average, a free will occur after
  // every 4th allocation
  constexpr std::size_t FREE_FREQUENCY{4};
  std::uniform_int_distribution<int> free_distribution(1, FREE_FREQUENCY);

  std::deque<allocation> allocations;

  constexpr std::size_t num_allocations{100};
  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t allocation_size = size_distribution(generator);
    EXPECT_NO_THROW(allocations.emplace_back(
        this->mr->allocate(allocation_size), allocation_size));
    auto new_allocation = allocations.back();
    EXPECT_NE(nullptr, new_allocation.p);
    EXPECT_TRUE(is_aligned(new_allocation.p));

    bool const free_front{free_distribution(generator) ==
                          free_distribution.max()};

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

TYPED_TEST(MRTest, MixedRandomAllocationFreeStream) {
  std::default_random_engine generator;

  constexpr std::size_t MAX_ALLOCATION_SIZE{10 * size_mb};
  std::uniform_int_distribution<std::size_t> size_distribution(
      1, MAX_ALLOCATION_SIZE);

  // How often a free will occur. For example, if `1`, then every allocation
  // will immediately be free'd. Or, if 4, on average, a free will occur after
  // every 4th allocation
  constexpr std::size_t FREE_FREQUENCY{4};
  std::uniform_int_distribution<int> free_distribution(1, FREE_FREQUENCY);

  std::deque<allocation> allocations;

  constexpr std::size_t num_allocations{100};
  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t allocation_size = size_distribution(generator);
    EXPECT_NO_THROW(allocations.emplace_back(
        this->mr->allocate(allocation_size, this->stream), allocation_size));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
    auto new_allocation = allocations.back();
    EXPECT_NE(nullptr, new_allocation.p);
    EXPECT_TRUE(is_aligned(new_allocation.p));

    bool const free_front{free_distribution(generator) ==
                          free_distribution.max()};

    if (free_front) {
      auto front = allocations.front();
      EXPECT_NO_THROW(this->mr->deallocate(front.p, front.size, this->stream));
      allocations.pop_front();
    }
  }
  // free any remaining allocations
  for (auto a : allocations) {
    EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size, this->stream));
    allocations.pop_front();
  }
}
