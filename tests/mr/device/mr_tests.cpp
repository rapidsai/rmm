/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

INSTANTIATE_TEST_SUITE_P(ResourceTests,
                         mr_test,
                         ::testing::Values(mr_factory{"CUDA", &make_cuda},
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
                                           mr_factory{"CUDA_Async", &make_cuda_async},
#endif
                                           mr_factory{"Managed", &make_managed},
                                           mr_factory{"Pool", &make_pool},
                                           mr_factory{"Arena", &make_arena},
                                           mr_factory{"Binning", &make_binning},
                                           mr_factory{"Fixed_Size", &make_fixed_size}),
                         [](auto const& info) { return info.param.name; });

// Leave out fixed-size MR here because it can't handle the dynamic allocation sizes
INSTANTIATE_TEST_SUITE_P(ResourceAllocationTests,
                         mr_allocation_test,
                         ::testing::Values(mr_factory{"CUDA", &make_cuda},
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
                                           mr_factory{"CUDA_Async", &make_cuda_async},
#endif
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
  auto* mr = rmm::mr::get_current_device_resource();
  EXPECT_NE(nullptr, mr);
  EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST_P(mr_test, SetCurrentDeviceResource)
{
  rmm::mr::device_memory_resource* old{};
  old = rmm::mr::set_current_device_resource(this->mr.get());
  EXPECT_NE(nullptr, old);

  // old mr should equal a cuda mr
  EXPECT_TRUE(old->is_equal(rmm::mr::cuda_memory_resource{}));

  // current dev resource should equal this resource
  EXPECT_TRUE(this->mr->is_equal(*rmm::mr::get_current_device_resource()));

  test_get_current_device_resource();

  // setting to `nullptr` should reset to initial cuda resource
  rmm::mr::set_current_device_resource(nullptr);
  EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST_P(mr_test, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TEST_P(mr_test, SupportsStreams)
{
  if (this->mr->is_equal(rmm::mr::cuda_memory_resource{}) ||
      this->mr->is_equal(rmm::mr::managed_memory_resource{})) {
    EXPECT_FALSE(this->mr->supports_streams());
  } else {
    EXPECT_TRUE(this->mr->supports_streams());
  }
}

TEST_P(mr_test, GetMemInfo)
{
  if (this->mr->supports_get_mem_info()) {
    const auto allocation_size{16 * 256};
    {
      auto const [free, total] = this->mr->get_mem_info(rmm::cuda_stream_view{});
      EXPECT_TRUE(free >= allocation_size);
    }

    void* ptr{nullptr};
    ptr = this->mr->allocate(allocation_size);

    {
      auto const [free, total] = this->mr->get_mem_info(rmm::cuda_stream_view{});
      EXPECT_TRUE(free >= allocation_size);
    }

    this->mr->deallocate(ptr, allocation_size);
  } else {
    auto const [free, total] = this->mr->get_mem_info(rmm::cuda_stream_view{});
    EXPECT_EQ(free, 0);
    EXPECT_EQ(total, 0);
  }
}

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TEST_P(mr_test, AllocationsAreDifferentDefaultStream)
{
  concurrent_allocations_are_different(this->mr.get(), cuda_stream_view{});
}

TEST_P(mr_test, AllocationsAreDifferent)
{
  concurrent_allocations_are_different(this->mr.get(), this->stream);
}

TEST_P(mr_allocation_test, AllocateDefaultStream)
{
  test_various_allocations(this->mr.get(), cuda_stream_view{});
}

TEST_P(mr_allocation_test, AllocateOnStream)
{
  test_various_allocations(this->mr.get(), this->stream);
}

TEST_P(mr_allocation_test, RandomAllocations) { test_random_allocations(this->mr.get()); }

TEST_P(mr_allocation_test, RandomAllocationsStream)
{
  test_random_allocations(this->mr.get(), default_num_allocations, default_max_size, this->stream);
}

TEST_P(mr_allocation_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->mr.get(), default_max_size, cuda_stream_view{});
}

TEST_P(mr_allocation_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_allocation_free(this->mr.get(), default_max_size, this->stream);
}

}  // namespace
}  // namespace rmm::test
