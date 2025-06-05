/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "mr_ref_test.hpp"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

INSTANTIATE_TEST_SUITE_P(ResourceTests,
                         mr_ref_test,
                         ::testing::Values("CUDA",
                                           "CUDA_Async",
                                           "Managed",
                                           "System",
                                           "Pool",
                                           "HostPinnedPool",
                                           "Arena",
                                           "Binning",
                                           "Fixed_Size"),
                         [](auto const& info) { return info.param; });

// Leave out fixed-size MR here because it can't handle the dynamic allocation sizes
INSTANTIATE_TEST_SUITE_P(ResourceAllocationTests,
                         mr_ref_allocation_test,
                         ::testing::Values("CUDA",
                                           "CUDA_Async",
                                           "Managed",
                                           "System"
                                           "Pool",
                                           "HostPinnedPool",
                                           "Arena",
                                           "Binning"),
                         [](auto const& info) { return info.param; });

TEST(DefaultTest, CurrentDeviceResourceIsCUDA)
{
  EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
  EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST(DefaultTest, UseCurrentDeviceResource) { test_get_current_device_resource(); }

TEST(DefaultTest, UseCurrentDeviceResourceRef) { test_get_current_device_resource_ref(); }

TEST(DefaultTest, GetCurrentDeviceResource)
{
  auto* mr = rmm::mr::get_current_device_resource();
  EXPECT_NE(nullptr, mr);
  EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST(DefaultTest, GetCurrentDeviceResourceRef)
{
  auto mr = rmm::mr::get_current_device_resource_ref();
  EXPECT_EQ(mr, rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
}

TEST_P(mr_ref_test, SetCurrentDeviceResourceRef)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  auto cuda_ref = rmm::device_async_resource_ref{cuda_mr};

  rmm::mr::set_current_device_resource_ref(cuda_ref);
  auto old = rmm::mr::set_current_device_resource_ref(this->ref);

  // old mr should equal a cuda mr
  EXPECT_EQ(old, cuda_ref);

  // current dev resource should equal this resource
  EXPECT_EQ(this->ref, rmm::mr::get_current_device_resource_ref());

  test_get_current_device_resource_ref();

  // Resetting should reset to initial cuda resource
  rmm::mr::reset_current_device_resource_ref();
  EXPECT_EQ(rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()},
            rmm::mr::get_current_device_resource_ref());
}

TEST_P(mr_ref_test, SelfEquality) { EXPECT_TRUE(this->ref == this->ref); }

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
TEST_P(mr_ref_test, AllocationsAreDifferent) { concurrent_allocations_are_different(this->ref); }

TEST_P(mr_ref_test, AsyncAllocationsAreDifferentDefaultStream)
{
  concurrent_async_allocations_are_different(this->ref, cuda_stream_view{});
}

TEST_P(mr_ref_test, AsyncAllocationsAreDifferent)
{
  concurrent_async_allocations_are_different(this->ref, this->stream);
}

TEST_P(mr_ref_allocation_test, AllocateDefault) { test_various_allocations(this->ref); }

TEST_P(mr_ref_allocation_test, AllocateDefaultStream)
{
  test_various_async_allocations(this->ref, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, AllocateOnStream)
{
  test_various_async_allocations(this->ref, this->stream);
}

TEST_P(mr_ref_allocation_test, RandomAllocations) { test_random_allocations(this->ref); }

TEST_P(mr_ref_allocation_test, RandomAllocationsDefaultStream)
{
  test_random_async_allocations(
    this->ref, default_num_allocations, default_max_size, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, RandomAllocationsStream)
{
  test_random_async_allocations(this->ref, default_num_allocations, default_max_size, this->stream);
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->ref, default_max_size);
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFreeDefaultStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, this->stream);
}

}  // namespace
}  // namespace rmm::test
