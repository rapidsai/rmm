/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/reduce.h>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::thrust_allocator<int>;

namespace rmm::test {
namespace {

struct allocator_test : public mr_ref_test {};

// Disable until we support resource_ref with set_current_device_resource
TEST_P(allocator_test, first)
{
  rmm::mr::set_current_device_resource_ref(this->ref);
  auto const num_ints{100};
  rmm::device_vector<int> ints(num_ints, 1);
  EXPECT_EQ(num_ints, thrust::reduce(ints.begin(), ints.end()));
}

TEST_P(allocator_test, defaults)
{
  rmm::mr::set_current_device_resource_ref(this->ref);
  rmm::mr::thrust_allocator<int> allocator(rmm::cuda_stream_default);
  EXPECT_EQ(allocator.stream(), rmm::cuda_stream_default);
  EXPECT_EQ(allocator.get_upstream_resource(),
            rmm::device_async_resource_ref{rmm::mr::get_current_device_resource_ref()});
}

TEST_P(allocator_test, multi_device)
{
  if (rmm::get_num_cuda_devices() < 2) { GTEST_SKIP() << "Needs at least two devices"; }
  cuda_set_device_raii with_device{rmm::get_current_cuda_device()};
  rmm::cuda_stream stream{};
  // make allocator on device-0
  rmm::mr::thrust_allocator<int> allocator(stream.view(), this->ref);
  auto const size{100};
  EXPECT_NO_THROW([&]() {
    auto vec = rmm::device_vector<int>(size, allocator);
    // Destruct with device-1 active
    RMM_CUDA_TRY(cudaSetDevice(1));
  }());
}

INSTANTIATE_TEST_SUITE_P(
  ThrustAllocatorTests,
  allocator_test,
  ::testing::Values("CUDA", "CUDA_Async", "Managed", "Pool", "Arena", "Binning"),
  [](auto const& info) { return info.param; });

}  // namespace
}  // namespace rmm::test
