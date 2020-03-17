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

#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/detail/contiguous_storage.h>

template <typename MR>
struct AllocatorTest : public ::testing::Test {
  MR mr{};
  cudaStream_t stream{};

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };
};

using resources = ::testing::Types<
    rmm::mr::cuda_memory_resource, rmm::mr::managed_memory_resource,
    rmm::mr::cnmem_memory_resource, rmm::mr::cnmem_managed_memory_resource>;
    //rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>;

TYPED_TEST_CASE(AllocatorTest, resources);

TYPED_TEST(AllocatorTest, first) {
    rmm::device_vector<int> ints(100);
}
