/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <rmm/error.hpp>
#include <rmm/mr/device/sam_headroom_memory_resource.hpp>
#include <rmm/mr/device/system_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace rmm::test {
namespace {

std::size_t constexpr size_mb{1_MiB};
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

// static property checks
static_assert(rmm::detail::polyfill::resource_with<system_mr, cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::async_resource_with<system_mr, cuda::mr::device_accessible>);

using headroom_mr = rmm::mr::sam_headroom_memory_resource;

// static property checks
static_assert(rmm::detail::polyfill::resource_with<headroom_mr, cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::async_resource_with<headroom_mr, cuda::mr::device_accessible>);

class SystemMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::mr::detail::is_system_memory_supported(rmm::get_current_cuda_device())) {
      GTEST_SKIP() << "Skipping tests since system memory allocator not supported with this "
                      "hardware/software version";
    }
  }
};

TEST(SystemMRSimpleTest, ThrowIfNotSupported)
{
  auto construct_mr = []() { system_mr mr; };
  if (rmm::mr::detail::is_system_memory_supported(rmm::get_current_cuda_device())) {
    EXPECT_NO_THROW(construct_mr());
  } else {
    EXPECT_THROW(construct_mr(), rmm::logic_error);
  }
}

TEST_F(SystemMRTest, FirstTouchOnCPU)
{
  auto const free = rmm::available_device_memory().first;
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_on_cpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_EQ(free, free2);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, FirstTouchOnGPU)
{
  auto const free = rmm::available_device_memory().first;
  system_mr mr;
  void* ptr = mr.allocate(size_mb);
  touch_on_gpu(ptr, size_mb);
  auto const free2 = rmm::available_device_memory().first;
  EXPECT_LT(free2, free);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, HeadroomMRReserveAllFreeMemory)
{
  auto const free = rmm::available_device_memory().first;
  // All the free GPU memory is set as headroom, so allocation is only on the CPU.
  headroom_mr mr{free + size_gb};
  void* ptr = mr.allocate(size_mb);
  touch_on_cpu(ptr, size_mb);
  mr.deallocate(ptr, size_mb);
}

TEST_F(SystemMRTest, HeadroomMRDifferentParametersUnequal)
{
  headroom_mr mr1{size_mb};
  headroom_mr mr2{size_gb};
  EXPECT_FALSE(mr1.is_equal(mr2));
}
}  // namespace
}  // namespace rmm::test
