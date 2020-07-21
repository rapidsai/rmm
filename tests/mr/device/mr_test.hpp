/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#pragma once

#include "gtest/gtest.h"

#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_multisize_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <random>

namespace rmm {
namespace test {

inline bool is_aligned(void* p, std::size_t alignment = 256)
{
  return (0 == reinterpret_cast<uintptr_t>(p) % alignment);
}

/**
 * @brief Returns if a pointer points to a device memory or managed memory
 * allocation.
 */
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

// some useful allocation sizes
constexpr long operator""_B(unsigned long long const x) { return x; }
constexpr long operator""_KiB(unsigned long long const x) { return x * (long{1} << 10); }
constexpr long operator""_MiB(unsigned long long const x) { return x * (long{1} << 20); }
constexpr long operator""_GiB(unsigned long long const x) { return x * (long{1} << 30); }
constexpr long operator""_TiB(unsigned long long const x) { return x * (long{1} << 40); }
constexpr long operator""_PiB(unsigned long long const x) { return x * (long{1} << 50); }

struct allocation {
  void* p{nullptr};
  std::size_t size{0};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};

// nested MR type names can get long...
using pool_mr       = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
using fixed_size_mr = rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>;
using fixed_multisize_mr =
  rmm::mr::fixed_multisize_memory_resource<rmm::mr::device_memory_resource>;
using fixed_multisize_pool_mr = rmm::mr::fixed_multisize_memory_resource<pool_mr>;
using hybrid_mr               = rmm::mr::hybrid_memory_resource<fixed_multisize_pool_mr, pool_mr>;

// Various test functions, shared between single-threaded and multithreaded tests.

inline void test_get_default_resource()
{
  EXPECT_NE(nullptr, rmm::mr::get_default_resource());
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_default_resource()->allocate(1_MiB));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_default_resource()->deallocate(p, 1_MiB));
}

template <typename MemoryResourceType>
void test_allocate(MemoryResourceType* mr, std::size_t bytes, cudaStream_t stream = 0)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = mr->allocate(bytes));
  if (stream != 0) EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(mr->deallocate(p, bytes));
  if (stream != 0) EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
}

void test_various_allocations(rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // test allocating zero bytes on non-default stream
  {
    void* p{nullptr};
    EXPECT_NO_THROW(p = mr->allocate(0, stream));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_NO_THROW(mr->deallocate(p, 0, stream));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  }

  test_allocate(mr, 4_B, stream);
  test_allocate(mr, 1_KiB, stream);
  test_allocate(mr, 1_MiB, stream);
  test_allocate(mr, 1_GiB, stream);

  // should fail to allocate too much
  {
    void* p{nullptr};
    EXPECT_THROW(p = mr->allocate(1_PiB, stream), rmm::bad_alloc);
    EXPECT_EQ(nullptr, p);
  }
}

void test_random_allocations(rmm::mr::device_memory_resource* mr,
                             std::size_t num_allocations = 100,
                             std::size_t max_size        = 5_MiB,
                             cudaStream_t stream         = 0)
{
  std::vector<allocation> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, max_size);

  // 100 allocations from [0,5MB)
  std::for_each(
    allocations.begin(), allocations.end(), [&generator, &distribution, stream, mr](allocation& a) {
      a.size = distribution(generator);
      EXPECT_NO_THROW(a.p = mr->allocate(a.size, stream));
      if (stream != 0) EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
      EXPECT_NE(nullptr, a.p);
      EXPECT_TRUE(is_aligned(a.p));
    });

  std::for_each(
    allocations.begin(), allocations.end(), [generator, distribution, stream, mr](allocation& a) {
      EXPECT_NO_THROW(mr->deallocate(a.p, a.size, stream));
      if (stream != 0) EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    });
}

template <typename MemoryResourceType>
void test_mixed_random_allocation_free(MemoryResourceType* mr,
                                       std::size_t max_size = 5_MiB,
                                       cudaStream_t stream  = 0)
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  constexpr int allocation_probability = 53;  // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations - 1);

  int active_allocations{0};
  int allocation_count{0};

  std::vector<allocation> allocations;

  for (int i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc   = (chance < allocation_probability) && (allocation_count < num_allocations);
    }

    if (do_alloc) {
      size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      EXPECT_NO_THROW(allocations.emplace_back(mr->allocate(size, stream), size));
      auto new_allocation = allocations.back();
      EXPECT_NE(nullptr, new_allocation.p);
      EXPECT_TRUE(is_aligned(new_allocation.p));
    } else {
      size_t index = index_distribution(generator) % active_allocations;
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      EXPECT_NO_THROW(mr->deallocate(to_free.p, to_free.size, stream));
    }
  }

  EXPECT_EQ(active_allocations, 0);
  EXPECT_EQ(allocations.size(), active_allocations);
}

// The test fixture
template <typename MemoryResourceType>
struct MRTest : public ::testing::Test {
  std::unique_ptr<MemoryResourceType> mr;
  cudaStream_t stream;

  MRTest() : mr{new MemoryResourceType} {}

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override { EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream)); };

  ~MRTest() {}
};

// Specialize constructor to pass arguments
template <>
inline MRTest<fixed_size_mr>::MRTest() : mr{new fixed_size_mr{rmm::mr::get_default_resource()}}
{
}

template <>
inline MRTest<fixed_multisize_mr>::MRTest()
  : mr{new fixed_multisize_mr(rmm::mr::get_default_resource())}
{
}

template <>
inline MRTest<pool_mr>::MRTest() : mr{}
{
  rmm::mr::cuda_memory_resource* cuda = new rmm::mr::cuda_memory_resource{};
  this->mr.reset(new pool_mr(cuda));
}

template <>
inline MRTest<hybrid_mr>::MRTest()
{
  rmm::mr::cuda_memory_resource* cuda = new rmm::mr::cuda_memory_resource{};
  pool_mr* pool                       = new pool_mr(cuda);
  this->mr.reset(new hybrid_mr(new fixed_multisize_pool_mr(pool), pool));
}

template <>
inline MRTest<pool_mr>::~MRTest()
{
  auto upstream = this->mr->get_upstream();
  this->mr.reset();
  delete upstream;
}

template <>
inline MRTest<hybrid_mr>::~MRTest()
{
  auto fixed = this->mr->get_small_mr();
  auto pool  = this->mr->get_large_mr();
  auto cuda  = pool->get_upstream();
  this->mr.reset();
  delete fixed;
  delete pool;
  delete cuda;
}

}  // namespace test
}  // namespace rmm
