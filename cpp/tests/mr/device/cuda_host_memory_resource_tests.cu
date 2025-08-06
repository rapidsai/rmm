/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device/cuda_host_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <random>

namespace rmm::test {
namespace {

std::size_t constexpr size_kb{1_KiB};
std::size_t constexpr size_mb{1_MiB};

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

void touch_on_cpu(void* ptr, std::size_t size)
{
  auto* data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i);
  }
}

class CudaHostMemoryResourceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Ensure CUDA is initialized
    cudaFree(nullptr);
  }
};

TEST_F(CudaHostMemoryResourceTest, BasicAllocation)
{
  rmm::mr::cuda_host_memory_resource mr;

  // Test basic allocation
  void* ptr = mr.allocate(size_kb);
  EXPECT_NE(nullptr, ptr);

  // Verify it's pinned host memory
  cudaPointerAttributes attributes{};
  EXPECT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_NE(nullptr, attributes.hostPointer);

  mr.deallocate(ptr, size_kb);
}

TEST_F(CudaHostMemoryResourceTest, ZeroSizeAllocation)
{
  rmm::mr::cuda_host_memory_resource mr;

  // Test zero size allocation
  void* ptr = mr.allocate(0);
  EXPECT_EQ(nullptr, ptr);

  mr.deallocate(ptr, 0);
}

TEST_F(CudaHostMemoryResourceTest, LargeAllocation)
{
  rmm::mr::cuda_host_memory_resource mr;

  // Test large allocation
  void* ptr = mr.allocate(size_mb);
  EXPECT_NE(nullptr, ptr);

  // Verify it's pinned host memory
  cudaPointerAttributes attributes{};
  EXPECT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_NE(nullptr, attributes.hostPointer);

  mr.deallocate(ptr, size_mb);
}

TEST_F(CudaHostMemoryResourceTest, MultipleAllocations)
{
  rmm::mr::cuda_host_memory_resource mr;

  std::vector<void*> ptrs;
  std::vector<std::size_t> sizes = {size_kb, size_kb * 2, size_kb * 4, size_kb * 8};

  // Allocate multiple blocks
  for (auto size : sizes) {
    void* ptr = mr.allocate(size);
    EXPECT_NE(nullptr, ptr);
    ptrs.push_back(ptr);
  }

  // Verify all are pinned host memory
  for (auto ptr : ptrs) {
    cudaPointerAttributes attributes{};
    EXPECT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
    EXPECT_NE(nullptr, attributes.hostPointer);
  }

  // Deallocate all
  for (std::size_t i = 0; i < ptrs.size(); ++i) {
    mr.deallocate(ptrs[i], sizes[i]);
  }
}

TEST_F(CudaHostMemoryResourceTest, AsyncAllocation)
{
  rmm::mr::cuda_host_memory_resource mr;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Test async allocation
  void* ptr = mr.allocate_async(size_kb, stream);
  EXPECT_NE(nullptr, ptr);

  // Verify it's pinned host memory
  cudaPointerAttributes attributes{};
  EXPECT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_NE(nullptr, attributes.hostPointer);

  mr.deallocate_async(ptr, size_kb, stream);

  cudaStreamDestroy(stream);
}

TEST_F(CudaHostMemoryResourceTest, CpuAccess)
{
  rmm::mr::cuda_host_memory_resource mr;

  void* ptr = mr.allocate(size_kb);
  EXPECT_NE(nullptr, ptr);

  // Test CPU access
  touch_on_cpu(ptr, size_kb);

  // Verify the data was written
  auto* data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size_kb; ++i) {
    EXPECT_EQ(static_cast<char>(i), data[i]);
  }

  mr.deallocate(ptr, size_kb);
}

TEST_F(CudaHostMemoryResourceTest, GpuAccess)
{
  rmm::mr::cuda_host_memory_resource mr;

  void* ptr = mr.allocate(size_kb);
  EXPECT_NE(nullptr, ptr);

  // Test GPU access
  touch_on_gpu(ptr, size_kb);

  // Verify the data was written by GPU
  auto* data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size_kb; ++i) {
    EXPECT_EQ(static_cast<char>(i), data[i]);
  }

  mr.deallocate(ptr, size_kb);
}

TEST_F(CudaHostMemoryResourceTest, CpuGpuRoundTrip)
{
  rmm::mr::cuda_host_memory_resource mr;

  void* ptr = mr.allocate(size_kb);
  EXPECT_NE(nullptr, ptr);

  // Write from CPU
  touch_on_cpu(ptr, size_kb);

  // Verify the data was written by CPU
  auto* data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size_kb; ++i) {
    EXPECT_EQ(static_cast<char>(i), data[i]);
  }

  // Read/write from GPU
  touch_on_gpu(ptr, size_kb);

  // Verify final state
  data = static_cast<char*>(ptr);
  for (std::size_t i = 0; i < size_kb; ++i) {
    EXPECT_EQ(static_cast<char>(i), data[i]);
  }

  mr.deallocate(ptr, size_kb);
}

TEST_F(CudaHostMemoryResourceTest, Equality)
{
  rmm::mr::cuda_host_memory_resource mr1;
  rmm::mr::cuda_host_memory_resource mr2;

  // Two instances should be equal
  EXPECT_TRUE(mr1.is_equal(mr2));
  EXPECT_TRUE(mr2.is_equal(mr1));

  // Self equality
  EXPECT_TRUE(mr1.is_equal(mr1));
}

TEST_F(CudaHostMemoryResourceTest, InequalityWithOtherTypes)
{
  rmm::mr::cuda_host_memory_resource host_mr;
  rmm::mr::cuda_memory_resource device_mr;

  // Should not be equal to device memory resource
  EXPECT_FALSE(host_mr.is_equal(device_mr));
  EXPECT_FALSE(device_mr.is_equal(host_mr));
}

TEST_F(CudaHostMemoryResourceTest, MemoryAlignment)
{
  rmm::mr::cuda_host_memory_resource mr;

  // Test various allocation sizes to check alignment
  std::vector<std::size_t> sizes = {1, 8, 16, 32, 64, 128, 256, 512, 1024};

  for (auto size : sizes) {
    void* ptr = mr.allocate(size);
    EXPECT_NE(nullptr, ptr);

    // Check that pointer is properly aligned
    EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(ptr) % 256);

    mr.deallocate(ptr, size);
  }
}

TEST_F(CudaHostMemoryResourceTest, StressTest)
{
  rmm::mr::cuda_host_memory_resource mr;

  constexpr std::size_t num_iterations = 1000;
  constexpr std::size_t max_size       = size_kb;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> size_dist(1, max_size);

  for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
    std::size_t size = size_dist(gen);
    void* ptr        = mr.allocate(size);
    EXPECT_NE(nullptr, ptr);

    // Touch the memory
    std::memset(ptr, static_cast<int>(iteration & 0xFF), size);

    mr.deallocate(ptr, size);
  }
}

}  // namespace
}  // namespace rmm::test
