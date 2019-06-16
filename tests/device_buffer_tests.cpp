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

#include "gtest/gtest.h"

#include <rmm/mr/cnmem_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>
#include <cstddef>

template <typename MemoryResourceType>
struct DeviceBufferTest : public ::testing::Test {
  cudaStream_t stream;

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource,
                                   rmm::mr::managed_memory_resource,
                                   rmm::mr::cnmem_memory_resource>;

TYPED_TEST_CASE(DeviceBufferTest, resources);

TYPED_TEST(DeviceBufferTest, DefaultMR) {
  constexpr std::size_t size{100};
  rmm::device_buffer buff(size);
  EXPECT_NE(nullptr, buff.data());
}
