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

#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/sub_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>

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

// some useful allocation sizes
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
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  cudaStream_t stream;

  MRTest() : mr{new MemoryResourceType} {}

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };

  ~MRTest() = default;

  void test_allocate(std::size_t bytes, cudaStream_t stream = 0);
  void test_random_allocations_base(std::size_t num_allocations = 100,
                                    std::size_t max_size = 5 * size_mb,
                                    cudaStream_t stream = 0);
  void test_random_allocations(std::size_t num_allocations = 100,
                                          cudaStream_t stream = 0);
  void test_mixed_random_allocation_free_base(std::size_t max_size = 5 * size_mb,
                                              cudaStream_t stream = 0);
  void test_mixed_random_allocation_free(cudaStream_t stream = 0);
};

// Specialize constructor to pass arguments
template <>
MRTest<rmm::mr::fixed_size_memory_resource>::MRTest() : 
  mr{new rmm::mr::fixed_size_memory_resource(1<<20, 32<<20)} {}

template <typename MemoryResourceType>
void MRTest<MemoryResourceType>::test_allocate(std::size_t bytes,
                                               cudaStream_t stream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(bytes));
  if (stream != 0)
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(this->mr->deallocate(p, bytes));
  if (stream != 0)
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

template <>
void MRTest<rmm::mr::fixed_size_memory_resource>::test_allocate(std::size_t bytes,
                                                                cudaStream_t stream) {
  void* p{nullptr};
  auto mr = reinterpret_cast<rmm::mr::fixed_size_memory_resource*>(this->mr.get());
  if (bytes > mr->get_block_size()) {
    EXPECT_THROW(p = this->mr->allocate(bytes), std::bad_alloc);
  }
  else {
    EXPECT_NO_THROW(p = mr->allocate(bytes));
    if (stream != 0)
      EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
    EXPECT_NE(nullptr, p);
    EXPECT_TRUE(is_aligned(p));
    EXPECT_TRUE(is_device_memory(p));
    EXPECT_NO_THROW(mr->deallocate(p, bytes));
    if (stream != 0)
      EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  }
}

template <typename MemoryResourceType>
void MRTest<MemoryResourceType>::test_random_allocations_base(std::size_t num_allocations,
                                                              std::size_t max_size,
                                                              cudaStream_t stream) {
  std::vector<allocation> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, max_size);

  // 100 allocations from [0,5MB)
  std::for_each(
      allocations.begin(), allocations.end(),
      [&generator, &distribution, stream, this](allocation& a) {
        a.size = distribution(generator);
        EXPECT_NO_THROW(a.p = this->mr->allocate(a.size, stream));
        if (stream != 0)
          EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_NE(nullptr, a.p);
        EXPECT_TRUE(is_aligned(a.p));
      });

  std::for_each(
      allocations.begin(), allocations.end(),
       [generator, distribution, stream, this](allocation& a) {
        EXPECT_NO_THROW(this->mr->deallocate(a.p, a.size, stream));
        if (stream != 0)
          EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
      });
}

template <typename MemoryResourceType>
void MRTest<MemoryResourceType>::test_random_allocations(std::size_t num_allocations,
                                                         cudaStream_t stream) {
  return test_random_allocations_base(num_allocations, 5 * size_mb, stream);
}

template <>
void MRTest<rmm::mr::fixed_size_memory_resource>::test_random_allocations(std::size_t num_allocations,
                                                                          cudaStream_t stream) {
  return test_random_allocations_base(num_allocations, 1 * size_mb, stream);
}

template <typename MemoryResourceType>
void MRTest<MemoryResourceType>::test_mixed_random_allocation_free_base(std::size_t max_size,
                                                                        cudaStream_t stream)
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  constexpr int allocation_probability = 53; // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations-1);

  int active_allocations{0};
  int allocation_count{0};

  std::vector<allocation> allocations;

  for (int i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc = (chance < allocation_probability) && 
                 (allocation_count < num_allocations);
    }

    if (do_alloc) {
      size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      EXPECT_NO_THROW(allocations.emplace_back(this->mr->allocate(size, stream), size));
      auto new_allocation = allocations.back();
      EXPECT_NE(nullptr, new_allocation.p);
      EXPECT_TRUE(is_aligned(new_allocation.p));
    }
    else {
      size_t index = index_distribution(generator) % active_allocations;
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      EXPECT_NO_THROW(this->mr->deallocate(to_free.p, to_free.size, stream));
    }
  }

  EXPECT_EQ(active_allocations, 0);
  EXPECT_EQ(allocations.size(), active_allocations);
}

template <typename MemoryResourceType>
void MRTest<MemoryResourceType>::test_mixed_random_allocation_free(cudaStream_t stream) {
  test_mixed_random_allocation_free_base(5 * size_mb, stream); 
}

template <>
void MRTest<rmm::mr::fixed_size_memory_resource>::test_mixed_random_allocation_free(cudaStream_t stream) {
  test_mixed_random_allocation_free_base(size_mb, stream);
}

// Test on all memory resource classes
using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource,
                                   rmm::mr::cnmem_managed_memory_resource,
                                   rmm::mr::sub_memory_resource,
                                   rmm::mr::fixed_size_memory_resource,
                                   rmm::mr::hybrid_memory_resource>;

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
  // Not necessarily false, since two cuda_memory_resources are always equal
  //EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
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
  // Not necessarily false, since two cuda_memory_resources are always equal
  //EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
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
  this->test_allocate(size_word);
}

TYPED_TEST(MRTest, AllocateWordStream) {
  this->test_allocate(size_word, this->stream);
}

TYPED_TEST(MRTest, AllocateKB) {
  this->test_allocate(size_kb);
}

TYPED_TEST(MRTest, AllocateKBStream) {
  this->test_allocate(size_kb, this->stream);
}

TYPED_TEST(MRTest, AllocateMB) {
  this->test_allocate(size_mb);
}

TYPED_TEST(MRTest, AllocateMBStream) {
  this->test_allocate(size_mb, this->stream);
}

TYPED_TEST(MRTest, AllocateGB) {
  this->test_allocate(size_gb);
}

TYPED_TEST(MRTest, AllocateGBStream) {
  this->test_allocate(size_gb, this->stream);
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
  this->test_random_allocations();
}

TYPED_TEST(MRTest, RandomAllocationsStream) {
  this->test_random_allocations(100, this->stream);
}

TYPED_TEST(MRTest, MixedRandomAllocationFree)
{
  this->test_mixed_random_allocation_free();
}
  
TYPED_TEST(MRTest, MixedRandomAllocationFreeStream)
{
  this->test_mixed_random_allocation_free(this->stream);
}

TYPED_TEST(MRTest, GetMemInfo) {
  std::pair<std::size_t,std::size_t> mem_info;
  EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
  std::size_t allocation_size = 16 * 256;
  void * ptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
  EXPECT_TRUE(mem_info.first >= allocation_size);
  EXPECT_NO_THROW(this->mr->deallocate(ptr,allocation_size));
}
