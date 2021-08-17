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

#include <cuda/memory_resource>

#include <gtest/gtest.h>

namespace rmm {
namespace test {
namespace {

INSTANTIATE_TEST_CASE_P(
  ResourceTests,
  mr_test,
  ::testing::Values(
    mr_factory{"CUDA", &make_cuda},
    mr_factory{"Managed", &make_managed},
    //#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    //                                        mr_factory{"CUDA_Async", &make_cuda_async},
    //#endif

    mr_factory{"Pool", &make_pool}
    //                                        mr_factory{"Arena", &make_arena},
    //                                        mr_factory { "Binning", &make_binning }

    ),
  [](auto const& info) { return info.param.name; });

TEST(DefaultTest, UseCurrentDeviceResource) { test_get_current_device_resource(); }

TEST(DefaultTest, GetCurrentDeviceResource)
{
  auto cuda_mr = rmm::mr::cuda_memory_resource{};
  EXPECT_TRUE(rmm::mr::get_current_device_resource_view() ==
              cuda::resource_view<cuda::memory_access::device>{&cuda_mr});
}

TEST_P(mr_test, SetCurrentDeviceResource)
{
  device_resource_view old{};
  EXPECT_NO_THROW(old = rmm::mr::set_current_device_resource_view(this->mr_view));

  EXPECT_NE(device_resource_view{nullptr}, old);

  // old mr should equal a cuda mr
  // auto cuda_mr = rmm::mr::cuda_memory_resource{};
  // EXPECT_TRUE(old == cuda::view_resource(&cuda_mr));

  // current dev resource should equal this resource
  EXPECT_TRUE(this->mr_view == rmm::mr::get_current_device_resource_view());

  // test_get_current_device_resource();

  // setting to `nullptr` should reset to initial cuda resource
  /*EXPECT_NO_THROW(rmm::mr::set_current_device_resource_view(nullptr));
  EXPECT_TRUE(rmm::mr::get_current_device_resource_view() == cuda::view_resource(&cuda_mr));*/
}

TEST_P(mr_test, SelfEquality) { EXPECT_TRUE(this->mr_view == this->mr_view); }

TEST_P(mr_test, AllocateDefaultStream)
{
  test_various_allocations(this->mr_view, DEFAULT_ALIGNMENT, cuda_stream_view{});
}

TEST_P(mr_test, AllocateOnStream)
{
  test_various_allocations(this->mr_view, DEFAULT_ALIGNMENT, this->stream);
}

TEST_P(mr_test, RandomAllocations) { test_random_allocations(this->mr_view); }

TEST_P(mr_test, RandomAllocationsStream)
{
  test_random_allocations(this->mr_view, 100, 5_MiB, this->stream);
}

TEST_P(mr_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->mr_view, 5_MiB, cuda_stream_view{});
}

TEST_P(mr_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_allocation_free(this->mr_view, 5_MiB, this->stream);
}

}  // namespace
}  // namespace test
}  // namespace rmm
