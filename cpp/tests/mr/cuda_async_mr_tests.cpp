/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {

using cuda_async_mr = rmm::mr::cuda_async_memory_resource;

// static property checks
static_assert(cuda::mr::synchronous_resource_with<cuda_async_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_async_mr, cuda::mr::device_accessible>);

class AsyncMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::runtime_async_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests since cudaMallocAsync not supported with this CUDA "
                   << "driver/runtime version";
    }
  }
};

TEST(RuntimeAsyncAllocTest, IsSupportedMatchesEachDevice)
{
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    int expected{};
    RMM_CUDA_TRY(cudaDeviceGetAttribute(&expected, cudaDevAttrMemoryPoolsSupported, device));
    EXPECT_EQ(rmm::detail::runtime_async_alloc::is_supported(rmm::cuda_device_id{device}),
              expected == 1);
  }
}

TEST(RuntimeAsyncAllocTest, IsSupportedConcurrentQueriesAgree)
{
  auto const num_devices = rmm::get_num_cuda_devices();
  std::vector<bool> expected(static_cast<std::size_t>(num_devices));
  for (int device = 0; device < num_devices; ++device) {
    int supported{};
    RMM_CUDA_TRY(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
    expected[static_cast<std::size_t>(device)] = supported == 1;
  }

  constexpr int num_threads{16};
  constexpr int iterations{1000};
  std::atomic<int> mismatches{};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread = 0; thread < num_threads; ++thread) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int device = 0; device < num_devices; ++device) {
          if (rmm::detail::runtime_async_alloc::is_supported(rmm::cuda_device_id{device}) !=
              expected[static_cast<std::size_t>(device)]) {
            ++mismatches;
          }
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(mismatches.load(), 0);
}

TEST(RuntimeAsyncAllocTest, IsSupportedInvalidDeviceIsFalse)
{
  EXPECT_FALSE(rmm::detail::runtime_async_alloc::is_supported(rmm::cuda_device_id{-1}));
  EXPECT_FALSE(rmm::detail::runtime_async_alloc::is_supported(
    rmm::cuda_device_id{rmm::get_num_cuda_devices()}));
}

TEST_F(AsyncMRTest, ExplicitInitialPoolSize)
{
  const auto pool_init_size{100};
  cuda_async_mr mr{pool_init_size};
  void* ptr = mr.allocate_sync(pool_init_size);
  mr.deallocate_sync(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncMRTest, ExplicitReleaseThreshold)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr{pool_init_size, pool_release_threshold};
  void* ptr = mr.allocate_sync(pool_init_size);
  mr.deallocate_sync(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncMRTest, DefaultReleaseThresholdIsUint64Max)
{
  cuda_async_mr mr{};
  std::uint64_t threshold{0};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
}

TEST_F(AsyncMRTest, ExplicitReleaseThresholdIsApplied)
{
  const std::uint64_t pool_release_threshold{1000};
  cuda_async_mr mr{{}, pool_release_threshold};
  std::uint64_t threshold{0};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, pool_release_threshold);
}

TEST_F(AsyncMRTest, DifferentPoolsUnequal)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr1{pool_init_size, pool_release_threshold};
  cuda_async_mr mr2{pool_init_size, pool_release_threshold};
  EXPECT_NE(mr1, mr2);
}

class AsyncMRFabricTest : public AsyncMRTest {
  void SetUp() override
  {
    AsyncMRTest::SetUp();

    auto handle_type = static_cast<cudaMemAllocationHandleType>(
      rmm::mr::cuda_async_memory_resource::allocation_handle_type::fabric);
    if (!rmm::detail::export_handle_type::is_supported(handle_type)) {
      GTEST_SKIP() << "Fabric handles are not supported in this environment. Skipping test.";
    }
  }
};

TEST_F(AsyncMRFabricTest, FabricHandlesSupport)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr{pool_init_size,
                   pool_release_threshold,
                   rmm::mr::cuda_async_memory_resource::allocation_handle_type::fabric};
  void* ptr = mr.allocate_sync(pool_init_size);
  mr.deallocate_sync(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
}  // namespace rmm::test
