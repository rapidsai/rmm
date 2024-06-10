/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "../../byte_literals.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/system_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

std::size_t constexpr size_mb{1_MiB};
std::size_t constexpr size_gb{1_GiB};
std::size_t constexpr size_2gb{2_GiB};

void touch_from_cpu(void* ptr, std::size_t size)
{
  auto* data              = static_cast<std::size_t*>(ptr);
  auto const num_elements = size / sizeof(std::size_t);
  for (std::size_t i = 0; i < num_elements; ++i) {
    data[i] = i;
  }
}

__global__ void touch_memory_kernel(std::size_t* data, std::size_t num_elements)
{
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_elements) { data[tid] = tid; }
}

void touch_from_gpu(void* ptr, std::size_t size)
{
  auto* data              = static_cast<std::size_t*>(ptr);
  auto const num_elements = size / sizeof(std::size_t);
  dim3 blockSize(256);
  dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
  touch_memory_kernel<<<gridSize, blockSize>>>(data, num_elements);
  cudaDeviceSynchronize();
}

using system_mr = rmm::mr::system_memory_resource;
static_assert(cuda::mr::resource_with<system_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::async_resource_with<system_mr, cuda::mr::device_accessible>);

class SystemMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::mr::detail::sam::is_supported()) {
      GTEST_SKIP() << "Skipping tests since system memory allocator not supported with this "
                      "hardware/software version";
    }
  }
};

TEST(SystemMRStandaloneTest, ThrowIfNotSupported)
{
  auto construct_mr = []() { system_mr mr; };
  if (rmm::mr::detail::sam::is_supported()) {
    EXPECT_NO_THROW(construct_mr());
  } else {
    EXPECT_THROW(construct_mr(), rmm::logic_error);
  }
}

TEST_F(SystemMRTest, PassthroughFirstTouchOnCPU)
{
  auto const [free, total] = rmm::available_device_memory();
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_from_cpu(ptr, size_mb);
  auto const [free2, total2] = rmm::available_device_memory();
  EXPECT_EQ(free, free2);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, PassthroughFirstTouchOnGPU)
{
  auto const [free, total] = rmm::available_device_memory();
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_from_gpu(ptr, size_mb);
  auto const [free2, total2] = rmm::available_device_memory();
  EXPECT_GT(free, free2);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, ExplicitHeadroom)
{
  auto const [free, total] = rmm::available_device_memory();
  system_mr mr{free, 0};
  // Since we set all the free memory as headroom, the pointer points to CPU memory.
  void* ptr                  = mr.allocate(size_gb);
  auto const [free2, total2] = rmm::available_device_memory();
  EXPECT_LE(free, free2);
  mr.deallocate(ptr, size_gb);
}

TEST_F(SystemMRTest, AboveThreshold)
{
  auto const [free, total] = rmm::available_device_memory();
  system_mr mr{free, size_gb};
  // Since we set all the free memory as headroom, and the allocation size is above the threshold,
  // the pointer points to CPU memory.
  void* ptr                  = mr.allocate(size_2gb);
  auto const [free2, total2] = rmm::available_device_memory();
  EXPECT_EQ(free, free2);
  mr.deallocate(ptr, size_gb);
}

TEST_F(SystemMRTest, DifferentParametersUnequal)
{
  system_mr mr1{size_gb, 0};
  system_mr mr2{size_2gb, 0};
  EXPECT_FALSE(mr1.is_equal(mr2));
}
}  // namespace
}  // namespace rmm::test
