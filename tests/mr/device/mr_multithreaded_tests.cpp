/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include "mr/device/cuda_memory_resource.hpp"
#include "mr/device/default_memory_resource.hpp"
#include "mr/device/pool_memory_resource.hpp"
#include "mr_test.hpp"

#include <thread>
#include <vector>

namespace {

using device_mr = rmm::mr::device_memory_resource;
using cuda_mr   = rmm::mr::cuda_memory_resource;
using thread_safe_fixed_size_mr =
  rmm::mr::thread_safe_resource_adaptor<fixed_size_mr, std::shared_ptr<fixed_size_mr>>;
using thread_safe_fixed_multisize_mr =
  rmm::mr::thread_safe_resource_adaptor<fixed_multisize_mr, std::shared_ptr<fixed_multisize_mr>>;
using thread_safe_fixed_multisize_pool_mr =
  rmm::mr::thread_safe_resource_adaptor<fixed_multisize_pool_mr,
                                        std::shared_ptr<fixed_multisize_pool_mr>>;
using thread_safe_hybrid_mr =
  rmm::mr::thread_safe_resource_adaptor<hybrid_mr, std::shared_ptr<hybrid_mr>>;

constexpr std::size_t num_threads{4};

template <typename Task, typename... Arguments>
void spawn(Task task, Arguments... args)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i)
    threads.emplace_back(std::thread(task, args...));

  for (auto& t : threads)
    t.join();
}

}  // namespace

// specialize test constructor for thread-safe types

template <>
inline MRTest<thread_safe_fixed_size_mr>::MRTest()
  : mr{new thread_safe_fixed_size_mr(
      std::make_shared<fixed_size_mr>(rmm::mr::get_default_resource()))}
{
}

template <>
inline MRTest<thread_safe_fixed_multisize_mr>::MRTest()
  : mr{new thread_safe_fixed_multisize_mr(
      std::make_shared<fixed_multisize_mr>(rmm::mr::get_default_resource()))}
{
}

template <>
inline MRTest<thread_safe_fixed_multisize_pool_mr>::MRTest()
  : mr{new thread_safe_fixed_multisize_pool_mr(std::make_shared<fixed_multisize_pool_mr>(
      std::make_shared<pool_mr>(std::make_shared<cuda_mr>())))}
{
}

template <>
inline MRTest<thread_safe_hybrid_mr>::MRTest()
{
  auto cuda = std::make_shared<cuda_mr>();
  auto pool = std::make_shared<pool_mr>(cuda);
  mr        = std::make_unique<thread_safe_hybrid_mr>(
    std::make_shared<hybrid_mr>(std::make_shared<fixed_multisize_pool_mr>(pool), pool));
}

// specialize get_max_size for thread-safe MRs
template <>
std::size_t get_max_size(thread_safe_fixed_size_mr* mr)
{
  return mr->get_upstream()->get_block_size();
}

template <>
std::size_t get_max_size(thread_safe_fixed_multisize_mr* mr)
{
  return mr->get_upstream()->get_max_size();
}

template <>
std::size_t get_max_size(thread_safe_fixed_multisize_pool_mr* mr)
{
  return mr->get_upstream()->get_max_size();
}

// specialize random allocations to not allocate too large
template <>
inline void test_random_allocations<thread_safe_fixed_size_mr>(thread_safe_fixed_size_mr* mr,
                                                               std::size_t num_allocations,
                                                               cudaStream_t stream)
{
  return test_random_allocations_base(mr, num_allocations, 1_MiB, stream);
}

template <>
inline void test_random_allocations<thread_safe_fixed_multisize_mr>(
  thread_safe_fixed_multisize_mr* mr, std::size_t num_allocations, cudaStream_t stream)
{
  return test_random_allocations_base(mr, num_allocations, 1_MiB, stream);
}

template <>
inline void test_random_allocations<thread_safe_fixed_multisize_pool_mr>(
  thread_safe_fixed_multisize_pool_mr* mr, std::size_t num_allocations, cudaStream_t stream)
{
  return test_random_allocations_base(mr, num_allocations, 1_MiB, stream);
}

template <>
inline void test_mixed_random_allocation_free<thread_safe_fixed_size_mr>(
  thread_safe_fixed_size_mr* mr, cudaStream_t stream)
{
  test_mixed_random_allocation_free_base(mr, 1_MiB, stream);
}

template <>
inline void test_mixed_random_allocation_free<thread_safe_fixed_multisize_mr>(
  thread_safe_fixed_multisize_mr* mr, cudaStream_t stream)
{
  test_mixed_random_allocation_free_base(mr, 4_MiB, stream);
}

template <>
inline void test_mixed_random_allocation_free<thread_safe_fixed_multisize_pool_mr>(
  thread_safe_fixed_multisize_pool_mr* mr, cudaStream_t stream)
{
  test_mixed_random_allocation_free_base(mr, 4_MiB, stream);
}

// Test on all memory resource classes
using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource,
                                   rmm::mr::cnmem_managed_memory_resource,
                                   pool_mr,
                                   thread_safe_fixed_size_mr,
                                   thread_safe_fixed_multisize_mr,
                                   thread_safe_fixed_multisize_pool_mr,
                                   thread_safe_hybrid_mr>;

template <typename MemoryResourceType>
using MRTest_mt = MRTest<MemoryResourceType>;

TYPED_TEST_CASE(MRTest_mt, resources);

TEST(DefaultTest, UseDefaultResource_mt) { spawn(test_get_default_resource); }

TYPED_TEST(MRTest_mt, SetDefaultResource_mt)
{
  // single thread changes default resource, then multiple threads use it

  // Not necessarily false, since two cuda_memory_resources are always equal
  // EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
  rmm::mr::device_memory_resource* old{nullptr};
  EXPECT_NO_THROW(old = rmm::mr::set_default_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);

  spawn([mr = this->mr.get()]() {
    EXPECT_EQ(mr, rmm::mr::get_default_resource());
    test_get_default_resource();  // test allocating with the new default resource
  });

  // setting default resource w/ nullptr should reset to initial
  EXPECT_NO_THROW(rmm::mr::set_default_resource(nullptr));
  EXPECT_TRUE(old->is_equal(*rmm::mr::get_default_resource()));
  // Not necessarily false, since two cuda_memory_resources are always equal
  // EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
}

TYPED_TEST(MRTest_mt, Allocate) { spawn(test_various_allocations<TypeParam>, this->mr.get()); }

TYPED_TEST(MRTest_mt, AllocateOnStream)
{
  spawn(test_various_allocations_on_stream<TypeParam>, this->mr.get(), this->stream);
}

TYPED_TEST(MRTest_mt, RandomAllocations)
{
  spawn(test_random_allocations<TypeParam>, this->mr.get(), 100, nullptr);
}

TYPED_TEST(MRTest_mt, RandomAllocationsStream)
{
  spawn(test_random_allocations<TypeParam>, this->mr.get(), 100, this->stream);
}

TYPED_TEST(MRTest_mt, MixedRandomAllocationFree)
{
  spawn(test_mixed_random_allocation_free<TypeParam>, this->mr.get(), nullptr);
}

TYPED_TEST(MRTest_mt, MixedRandomAllocationFreeStream)
{
  spawn(test_mixed_random_allocation_free<TypeParam>, this->mr.get(), this->stream);
}

template <typename MemoryResourceType>
void allocate_loop(MemoryResourceType* mr,
                   std::size_t num_allocations,
                   std::list<allocation>& allocations,
                   std::mutex& mtx,
                   cudaStream_t stream)
{
  constexpr std::size_t max_size{1_MiB};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  for (std::size_t i = 0; i < num_allocations; ++i) {
    size_t size = size_distribution(generator);
    void* ptr{};
    EXPECT_NO_THROW(ptr = mr->allocate(size, stream));
    {
      std::lock_guard<std::mutex> lock(mtx);
      allocations.emplace_back(ptr, size);
    }
  }
}

template <typename MemoryResourceType>
void deallocate_loop(MemoryResourceType* mr,
                     std::size_t num_allocations,
                     std::list<allocation>& allocations,
                     std::mutex& mtx,
                     cudaStream_t stream)
{
  for (std::size_t i = 0; i < num_allocations;) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (allocations.empty())
        continue;
      else {
        i++;
        allocation alloc = allocations.front();
        allocations.pop_front();
        EXPECT_NO_THROW(mr->deallocate(alloc.p, alloc.size, stream));
      }
    }
  }
}

template <typename MemoryResourceType>
void test_allocate_free_different_threads(MemoryResourceType* mr,
                                          cudaStream_t streamA,
                                          cudaStream_t streamB)
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};
  constexpr std::size_t max_size{1_MiB};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  std::mutex mtx;
  std::list<allocation> allocations;

  std::thread producer(allocate_loop<MemoryResourceType>,
                       mr,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       streamA);
  std::thread consumer(deallocate_loop<MemoryResourceType>,
                       mr,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       streamB);

  producer.join();
  consumer.join();
}

TYPED_TEST(MRTest_mt, AllocFreeDifferentThreadsDefaultStream)
{
  test_allocate_free_different_threads<TypeParam>(this->mr.get(), nullptr, nullptr);
}

TYPED_TEST(MRTest_mt, AllocFreeDifferentThreadsSameStream)
{
  test_allocate_free_different_threads<TypeParam>(this->mr.get(), this->stream, this->stream);
}

// cnmem does not allow freeing on a different stream than allocating
using resources_different_stream = ::testing::Types<rmm::mr::cuda_memory_resource,
                                                    rmm::mr::managed_memory_resource,
                                                    pool_mr,
                                                    thread_safe_fixed_size_mr,
                                                    thread_safe_fixed_multisize_mr,
                                                    thread_safe_fixed_multisize_pool_mr,
                                                    thread_safe_hybrid_mr>;

template <typename MemoryResourceType>
using MRTestDifferentStream_mt = MRTest<MemoryResourceType>;

TYPED_TEST_CASE(MRTestDifferentStream_mt, resources_different_stream);

TYPED_TEST(MRTestDifferentStream_mt, AllocFreeDifferentThreadsDifferentStream)
{
  cudaStream_t streamB{};
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&streamB));
  test_allocate_free_different_threads<TypeParam>(this->mr.get(), this->stream, streamB);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(streamB));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(streamB));
}
