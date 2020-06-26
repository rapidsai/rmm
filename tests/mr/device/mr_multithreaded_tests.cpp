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

#include "mr/device/cuda_memory_resource.hpp"
#include "mr/device/default_memory_resource.hpp"
#include "mr/device/pool_memory_resource.hpp"
#include "mr_test.hpp"

#include <thread>
#include <vector>

namespace {

using thread_safe_pool_mr            = rmm::mr::thread_safe_resource_adaptor<pool_mr>;
using thread_safe_fixed_size_mr      = rmm::mr::thread_safe_resource_adaptor<fixed_size_mr>;
using thread_safe_fixed_multisize_mr = rmm::mr::thread_safe_resource_adaptor<fixed_multisize_mr>;
using thread_safe_fixed_multisize_pool_mr =
  rmm::mr::thread_safe_resource_adaptor<fixed_multisize_pool_mr>;
using thread_safe_hybrid_mr = rmm::mr::thread_safe_resource_adaptor<hybrid_mr>;

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
inline MRTest<thread_safe_pool_mr>::MRTest()
  : mr{new thread_safe_pool_mr(new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
      new rmm::mr::cuda_memory_resource))}
{
}

template <>
inline MRTest<thread_safe_pool_mr>::~MRTest()
{
  auto pool = mr->get_upstream();
  auto cuda = pool->get_upstream();
  delete pool;
  delete cuda;
}

template <>
inline MRTest<thread_safe_fixed_size_mr>::MRTest()
  : mr{new thread_safe_fixed_size_mr(new fixed_size_mr(rmm::mr::get_default_resource()))}
{
}

template <>
inline MRTest<thread_safe_fixed_size_mr>::~MRTest()
{
  auto fixed = mr->get_upstream();
  this->mr.reset();
  delete fixed;
}

template <>
inline MRTest<thread_safe_fixed_multisize_mr>::MRTest()
  : mr{new thread_safe_fixed_multisize_mr(new fixed_multisize_mr(rmm::mr::get_default_resource()))}
{
}

template <>
inline MRTest<thread_safe_fixed_multisize_mr>::~MRTest()
{
  auto fixed = mr->get_upstream();
  this->mr.reset();
  delete fixed;
}

template <>
inline MRTest<thread_safe_fixed_multisize_pool_mr>::MRTest()
  : mr{new thread_safe_fixed_multisize_pool_mr(
      new fixed_multisize_pool_mr(new pool_mr(new rmm::mr::cuda_memory_resource)))}
{
}

template <>
inline MRTest<thread_safe_fixed_multisize_pool_mr>::~MRTest()
{
  auto fixed = mr->get_upstream();
  auto pool  = fixed->get_upstream();
  auto cuda  = pool->get_upstream();
  this->mr.reset();
  delete fixed;
  delete pool;
  delete cuda;
}

template <>
inline MRTest<thread_safe_hybrid_mr>::MRTest()
{
  rmm::mr::cuda_memory_resource* cuda = new rmm::mr::cuda_memory_resource{};
  pool_mr* pool                       = new pool_mr(cuda);
  this->mr.reset(new thread_safe_hybrid_mr(new hybrid_mr(new fixed_multisize_pool_mr(pool), pool)));
}

template <>
inline MRTest<thread_safe_hybrid_mr>::~MRTest()
{
  auto hybrid = mr->get_upstream();
  auto fixed  = hybrid->get_small_mr();
  auto pool   = hybrid->get_large_mr();
  auto cuda   = pool->get_upstream();
  this->mr.reset();
  delete hybrid;
  delete fixed;
  delete pool;
  delete cuda;
}

// Test on all memory resource classes
using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource,
                                   rmm::mr::cnmem_managed_memory_resource,
                                   thread_safe_pool_mr,
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
