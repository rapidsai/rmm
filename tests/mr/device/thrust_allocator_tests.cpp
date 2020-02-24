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

#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/sub_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>


#include <gtest/gtest.h>

template <typename MemoryResourceType>
struct AllocatorTest : public ::testing::Test {
  MemoryResourceType mr{};
  cudaStream_t stream{};

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource,
                                   rmm::mr::cnmem_managed_memory_resource,
                                   rmm::mr::sub_memory_resource<rmm::mr::cuda_memory_resource>>;

TYPED_TEST_CASE(AllocatorTest, resources);


// TODO Add tests once RMM cmake is updated to allow compiling .cu files
