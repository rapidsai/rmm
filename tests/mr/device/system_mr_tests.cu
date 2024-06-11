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
std::size_t constexpr size_2mb{2_MiB};
std::size_t constexpr size_gb{1_GiB};

void touch_on_cpu(void* ptr, std::size_t size)
{
  auto* data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i);
  }
}

__global__ void touch_memory_kernel(char* data, std::size_t size)
{
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { data[tid] = static_cast<char>(tid); }
}

void touch_on_gpu(void* ptr, std::size_t size)
{
  dim3 blockSize(256);
  dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
  touch_memory_kernel<<<gridSize, blockSize>>>(static_cast<char*>(ptr), size);
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
  auto const free = rmm::available_device_memory().first;
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_on_cpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_EQ(free, free2);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, PassthroughFirstTouchOnGPU)
{
  auto const free = rmm::available_device_memory().first;
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_on_gpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_LT(free2, free);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, ExplicitHeadroom)
{
  auto const free = rmm::available_device_memory().first;
  // All the free GPU memory is set as headroom, so allocation is only on the CPU.
  system_mr mr{free};
  void* ptr = mr.allocate(size_mb);
  touch_on_gpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_GE(free2, free);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, BelowThreshold)
{
  auto const free = rmm::available_device_memory().first;
  system_mr mr{size_gb, size_gb};
  void* ptr = mr.allocate(size_mb);
  touch_on_cpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_GE(free2, free);
  mr.deallocate(ptr, size_gb);
}

TEST_F(SystemMRTest, AboveThreshold)
{
  auto const free = rmm::available_device_memory().first;
  system_mr mr{size_mb, size_mb};
  void* ptr = mr.allocate(size_2mb);
  touch_on_gpu(ptr, size_2mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_LE(free2, free);
  mr.deallocate(ptr, size_2mb);
}

TEST_F(SystemMRTest, DifferentParametersUnequal)
{
  system_mr mr1{size_mb, 0};
  system_mr mr2{size_gb, 0};
  EXPECT_FALSE(mr1.is_equal(mr2));
}
}  // namespace
}  // namespace rmm::test
