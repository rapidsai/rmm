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

#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;

using resources =
    ::testing::Types<rmm::mr::cuda_memory_resource,
                     rmm::mr::managed_memory_resource,
                     rmm::mr::cnmem_memory_resource,
                     rmm::mr::cnmem_managed_memory_resource, pool_mr>;

template <typename MR>
struct AllocatorTest : public ::testing::Test {
  std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> upstreams;
  std::unique_ptr<MR> mr;

  AllocatorTest() : mr{std::make_unique<MR>()} {
    rmm::mr::set_default_resource(mr.get());
  }

  ~AllocatorTest() = default;
};

template <>
AllocatorTest<pool_mr>::AllocatorTest() {
  upstreams.emplace_back(std::make_unique<rmm::mr::cuda_memory_resource>());
  auto& cuda_upstream = upstreams.front();
  mr.reset(new pool_mr(static_cast<rmm::mr::cuda_memory_resource*>(cuda_upstream.get())));
  rmm::mr::set_default_resource(mr.get());
}

TYPED_TEST_CASE(AllocatorTest, resources);

TYPED_TEST(AllocatorTest, first) {
  rmm::device_vector<int> ints(100, 1);
  EXPECT_EQ(100, thrust::reduce(ints.begin(), ints.end()));
}
