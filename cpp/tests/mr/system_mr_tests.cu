/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/sam_headroom_memory_resource.hpp>
#include <rmm/mr/system_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace rmm::test {
namespace {

std::size_t constexpr size_mb{1_MiB};
std::size_t constexpr size_gb{1_GiB};
std::size_t constexpr size_64mb{64_MiB};

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
  RMM_CUDA_TRY(cudaGetLastError());
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

using system_mr = rmm::mr::system_memory_resource;

// static property checks
static_assert(cuda::mr::synchronous_resource_with<system_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<system_mr, cuda::mr::device_accessible>);

using headroom_mr = rmm::mr::sam_headroom_memory_resource;

// static property checks
static_assert(cuda::mr::synchronous_resource_with<headroom_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<headroom_mr, cuda::mr::device_accessible>);

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

// On ATS systems (e.g. Grace Hopper, Grace Blackwell) the GPU memory is a NUMA node, so the kernel
// places each system-allocated page on the NUMA node of the processor that first touches it.
// /proc/self/numa_maps lists the pages resident on each node (`N<node>=<pages>`) for each mapping;
// unlike move_pages(2) it is not blocked by container seccomp profiles. The kernel may merge an
// allocation's mapping with adjacent ones, so the counts can include pages outside the allocation.
std::map<int, std::size_t> numa_pages_of_mapping_containing(void* ptr)
{
  auto const addr = reinterpret_cast<std::uintptr_t>(ptr);
  std::ifstream numa_maps("/proc/self/numa_maps");
  std::map<int, std::size_t> pages;
  std::string line;
  while (std::getline(numa_maps, line)) {
    if (std::stoull(line, nullptr, 16) > addr) { break; }
    pages.clear();
    std::istringstream fields(line);
    std::string field;
    while (fields >> field) {
      auto const eq = field.find('=');
      if (field.size() > 1 && field[0] == 'N' && eq != std::string::npos) {
        pages[std::stoi(field.substr(1, eq - 1))] = std::stoull(field.substr(eq + 1));
      }
    }
  }
  return pages;
}

class SystemMRFirstTouchTest : public SystemMRTest {
 protected:
  void SetUp() override
  {
    SystemMRTest::SetUp();
    if (IsSkipped()) { return; }
    auto const device = rmm::get_current_cuda_device().value();
    int numa_config{};
    RMM_CUDA_TRY(cudaDeviceGetAttribute(&numa_config, cudaDevAttrNumaConfig, device));
    if (numa_config != cudaDeviceNumaConfigNumaNode) {
      GTEST_SKIP() << "Skipping first-touch placement tests: GPU memory is not a NUMA node";
    }
    RMM_CUDA_TRY(cudaDeviceGetAttribute(&gpu_numa_id, cudaDevAttrNumaId, device));
  }

  int gpu_numa_id{-1};
};

TEST_F(SystemMRFirstTouchTest, FirstTouchOnCPU)
{
  system_mr mr;
  void* ptr = mr.allocate_sync(size_64mb);
  touch_on_cpu(ptr, size_64mb);
  auto const pages = numa_pages_of_mapping_containing(ptr);
  ASSERT_FALSE(pages.empty());
  EXPECT_EQ(pages.count(gpu_numa_id), 0);
  mr.deallocate_sync(ptr, size_64mb);
}

TEST_F(SystemMRFirstTouchTest, FirstTouchOnGPU)
{
  system_mr mr;
  void* ptr = mr.allocate_sync(size_64mb);
  touch_on_gpu(ptr, size_64mb);
  auto const pages = numa_pages_of_mapping_containing(ptr);
  ASSERT_EQ(pages.count(gpu_numa_id), 1);
  auto const allocation_pages = size_64mb / static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  // The allocator's own bookkeeping touches the start of the allocation from the CPU.
  EXPECT_GE(pages.at(gpu_numa_id), allocation_pages * 9 / 10);
  mr.deallocate_sync(ptr, size_64mb);
}

TEST_F(SystemMRTest, HeadroomMRReserveAllFreeMemory)
{
  auto const free = rmm::available_device_memory().first;
  // All the free GPU memory is set as headroom, so allocation is only on the CPU.
  headroom_mr mr{free + size_gb};
  void* ptr = mr.allocate_sync(size_mb);
  touch_on_cpu(ptr, size_mb);
  mr.deallocate_sync(ptr, size_mb);
}

TEST_F(SystemMRTest, HeadroomMRDifferentParametersUnequal)
{
  headroom_mr mr1{size_mb};
  headroom_mr mr2{size_gb};
  EXPECT_NE(mr1, mr2);
}
}  // namespace
}  // namespace rmm::test
