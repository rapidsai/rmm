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

#include <rmm/mr/device/per_device_resource.hpp>
#include "mr_test.hpp"

#include <gtest/gtest.h>

namespace rmm {
namespace test {
namespace {

INSTANTIATE_TEST_CASE_P(ResourceTests,
                        mr_test,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Arena", &make_arena},
                                          mr_factory{"Binning", &make_binning}),
                        [](auto const& info) { return info.param.name; });

TEST(DefaultTest, CurrentDeviceResourceIsCUDA)
{
  EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
  EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST(DefaultTest, UseCurrentDeviceResource) { test_get_current_device_resource(); }

TEST(DefaultTest, GetCurrentDeviceResource)
{
  rmm::mr::device_memory_resource* mr;
  EXPECT_NO_THROW(mr = rmm::mr::get_current_device_resource());
  EXPECT_NE(nullptr, mr);
  EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST_P(mr_test, SetCurrentDeviceResource)
{
  rmm::mr::device_memory_resource* old{};
  EXPECT_NO_THROW(old = rmm::mr::set_current_device_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);

  // old mr should equal a cuda mr
  EXPECT_TRUE(old->is_equal(rmm::mr::cuda_memory_resource{}));

  // current dev resource should equal this resource
  EXPECT_TRUE(this->mr->is_equal(*rmm::mr::get_current_device_resource()));

  test_get_current_device_resource();

  // setting to `nullptr` should reset to initial cuda resource
  EXPECT_NO_THROW(rmm::mr::set_current_device_resource(nullptr));
  EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST_P(mr_test, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TEST_P(mr_test, AllocateDefaultStream)
{
  test_various_allocations(this->mr.get(), cudaStreamDefault);
}

TEST_P(mr_test, AllocateOnStream) { test_various_allocations(this->mr.get(), this->stream); }

TEST_P(mr_test, RandomAllocations) { test_random_allocations(this->mr.get()); }

TEST_P(mr_test, RandomAllocationsStream)
{
  test_random_allocations(this->mr.get(), 100, 5_MiB, this->stream);
}

TEST_P(mr_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->mr.get(), 5_MiB, cudaStreamDefault);
}

TEST_P(mr_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_allocation_free(this->mr.get(), 5_MiB, this->stream);
}

TEST_P(mr_test, GetMemInfo)
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
}  // namespace
}  // namespace test
}  // namespace rmm
