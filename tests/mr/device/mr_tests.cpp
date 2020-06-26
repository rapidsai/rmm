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

#include "mr_test.hpp"

// Test on all memory resource classes
using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource,
                                   rmm::mr::cnmem_managed_memory_resource,
                                   thread_safe_cuda_mr,
                                   pool_mr,
                                   fixed_size_mr,
                                   fixed_multisize_mr,
                                   hybrid_mr>;

TYPED_TEST_CASE(MRTest, resources);

TEST(DefaultTest, UseDefaultResource)
{
  EXPECT_NE(nullptr, rmm::mr::get_default_resource());
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_default_resource()->allocate(1_MiB));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_default_resource()->deallocate(p, 1_MiB));
}

TYPED_TEST(MRTest, SetDefaultResource)
{
  // Not necessarily false, since two cuda_memory_resources are always equal
  // EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
  rmm::mr::device_memory_resource* old{nullptr};
  EXPECT_NO_THROW(old = rmm::mr::set_default_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);
  EXPECT_TRUE(this->mr->is_equal(*rmm::mr::get_default_resource()));
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_default_resource()->allocate(1_MiB));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_default_resource()->deallocate(p, 1_MiB));
  // setting default resource w/ nullptr should reset to initial
  EXPECT_NO_THROW(rmm::mr::set_default_resource(nullptr));
  EXPECT_TRUE(old->is_equal(*rmm::mr::get_default_resource()));
  // Not necessarily false, since two cuda_memory_resources are always equal
  // EXPECT_FALSE(this->mr->is_equal(*rmm::mr::get_default_resource()));
}

TYPED_TEST(MRTest, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TYPED_TEST(MRTest, AllocateZeroBytes)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0));
  EXPECT_EQ(nullptr, p);
  EXPECT_NO_THROW(this->mr->deallocate(p, 0));
}

TYPED_TEST(MRTest, AllocateZeroBytesStream)
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = this->mr->allocate(0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
  EXPECT_NO_THROW(this->mr->deallocate(p, 0, this->stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream));
}

TYPED_TEST(MRTest, Allocate)
{
  this->test_allocate(4_B);
  this->test_allocate(1_KiB);
  this->test_allocate(1_MiB);
  this->test_allocate(1_GiB);

  // should fail to allocate too much
  void* p{nullptr};
  EXPECT_THROW(p = this->mr->allocate(1_PiB), rmm::bad_alloc);
  EXPECT_EQ(nullptr, p);
}

TYPED_TEST(MRTest, AllocateOnStream)
{
  this->test_allocate(4_B, this->stream);
  this->test_allocate(1_KiB, this->stream);
  this->test_allocate(1_MiB, this->stream);
  this->test_allocate(1_GiB, this->stream);

  // should fail to allocate too much
  void* p{nullptr};
  EXPECT_THROW(p = this->mr->allocate(1_PiB, this->stream), rmm::bad_alloc);
  EXPECT_EQ(nullptr, p);
}

TYPED_TEST(MRTest, RandomAllocations) { this->test_random_allocations(); }

TYPED_TEST(MRTest, RandomAllocationsStream) { this->test_random_allocations(100, this->stream); }

TYPED_TEST(MRTest, MixedRandomAllocationFree) { this->test_mixed_random_allocation_free(); }

TYPED_TEST(MRTest, MixedRandomAllocationFreeStream)
{
  this->test_mixed_random_allocation_free(this->stream);
}

TYPED_TEST(MRTest, GetMemInfo)
{
  if (this->mr->supports_get_mem_info()) {
    std::pair<std::size_t, std::size_t> mem_info;
    EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
    std::size_t allocation_size = 16 * 256;
    void* ptr;
    EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
    EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
    EXPECT_TRUE(mem_info.first >= allocation_size);
    EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
  }
}
