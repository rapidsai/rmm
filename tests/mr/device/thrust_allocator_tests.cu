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

#include <gtest/gtest.h>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include "mr_test.hpp"

namespace rmm {
namespace test {
namespace {

struct allocator_test : public mr_test {
};

TEST_P(allocator_test, first)
{
  rmm::device_vector<int> ints(100, 1);
  EXPECT_EQ(100, thrust::reduce(ints.begin(), ints.end()));
}

INSTANTIATE_TEST_CASE_P(ThrustAllocatorTests,
                        allocator_test,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Binning", &make_binning}),
                        [](auto const& info) { return info.param.name; });
}  // namespace
}  // namespace test
}  // namespace rmm
