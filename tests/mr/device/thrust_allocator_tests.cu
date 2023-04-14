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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <gtest/gtest.h>

#include <thrust/reduce.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::thrust_allocator<int>;

namespace rmm::test {
namespace {

struct allocator_test : public mr_test {};

TEST_P(allocator_test, first)
{
  auto const num_ints{100};
  rmm::device_vector<int> ints(num_ints, 1);
  EXPECT_EQ(num_ints, thrust::reduce(ints.begin(), ints.end()));
}

TEST_P(allocator_test, defaults)
{
  rmm::mr::thrust_allocator<int> allocator(rmm::cuda_stream_default);
  EXPECT_EQ(allocator.stream(), rmm::cuda_stream_default);
  EXPECT_EQ(allocator.resource(), rmm::mr::get_current_device_resource());
}

INSTANTIATE_TEST_CASE_P(ThrustAllocatorTests,
                        allocator_test,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
                                          mr_factory{"CUDA_Async", &make_cuda_async},
#endif
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Arena", &make_arena},
                                          mr_factory{"Binning", &make_binning}),
                        [](auto const& info) { return info.param.name; });

}  // namespace
}  // namespace rmm::test
