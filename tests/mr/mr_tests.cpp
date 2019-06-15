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

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cstddef>
#include <random>

namespace {
static constexpr std::size_t ALIGNMENT{256};
inline bool is_aligned(void* p, std::size_t alignment = ALIGNMENT) {
  return (0 == reinterpret_cast<uintptr_t>(p) % alignment);
}

inline bool is_device_memory(void* p) {
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, p)) {
    return false;
  }
  return attributes.memoryType == cudaMemoryTypeDevice;
}

static constexpr std::size_t size_word{4};
static constexpr std::size_t size_kb{std::size_t{1} << 10};
static constexpr std::size_t size_mb{std::size_t{1} << 20};
static constexpr std::size_t size_gb{std::size_t{1} << 30};
static constexpr std::size_t size_tb{std::size_t{1} << 40};
static constexpr std::size_t size_pb{std::size_t{1} << 50};
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

using resources = ::testing::Types<rmm::mr::cuda_memory_resource>;

TYPED_TEST_CASE(MRTest, resources);

TYPED_TEST(MRTest, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TYPED_TEST(MRTest, AllocateZeroBytesStream) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NO_THROW(this->mr->deallocate(p, 0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, AllocateZeroBytes) {
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0));
  EXPECT_EQ(nullptr, p);
  EXPECT_NO_THROW(this->mr->deallocate(p, 0));
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
  using allocation_t = std::pair<void*, std::size_t>;
  constexpr std::size_t num_allocations{100};
  std::vector<allocation_t> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 3 * size_mb);

  // 100 allocations from [0,3MB)
  std::for_each(allocations.begin(), allocations.end(),
                [generator, distribution, this](allocation_t& allocation) {
                  allocation.second = distribution(generator);
                  EXPECT_NO_THROW(allocation.first =
                                      this->mr->allocate(allocation.second));
                  EXPECT_NE(nullptr, allocation.first);
                });

  std::for_each(allocations.begin(), allocations.end(),
                [generator, distribution, this](allocation_t& allocation) {
                  EXPECT_NO_THROW(this->mr->deallocate(allocation.first,
                                                       allocation.second));
                });
}
