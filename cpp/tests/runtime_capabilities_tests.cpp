/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {

bool device_flag(cudaDeviceAttr attribute, int device)
{
  int value{};
  RMM_CUDA_TRY(cudaDeviceGetAttribute(&value, attribute, device));
  return value == 1;
}

bool async_managed_versions_supported()
{
  int driver_version{};
  int runtime_version{};
  RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
  RMM_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
  return driver_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION and
         runtime_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION;
}

TEST(RuntimeCapabilitiesTest, RuntimeAsyncAllocMatchesEachDevice)
{
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    EXPECT_EQ(rmm::detail::runtime_async_alloc::is_supported(rmm::cuda_device_id{device}),
              device_flag(cudaDevAttrMemoryPoolsSupported, device));
  }
}

TEST(RuntimeCapabilitiesTest, ConcurrentManagedAccessMatchesEachDevice)
{
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    EXPECT_EQ(rmm::detail::concurrent_managed_access::is_supported(rmm::cuda_device_id{device}),
              device_flag(cudaDevAttrConcurrentManagedAccess, device));
  }
}

TEST(RuntimeCapabilitiesTest, RuntimeAsyncManagedAllocMatchesEachDevice)
{
  auto const versions_supported = async_managed_versions_supported();
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    EXPECT_EQ(rmm::detail::runtime_async_managed_alloc::is_supported(rmm::cuda_device_id{device}),
              versions_supported and device_flag(cudaDevAttrConcurrentManagedAccess, device));
  }
}

TEST(RuntimeCapabilitiesTest, DeviceIntegratedMemoryMatchesEachDevice)
{
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    EXPECT_EQ(rmm::detail::device_integrated_memory::is_supported(rmm::cuda_device_id{device}),
              device_flag(cudaDevAttrIntegrated, device));
  }
}

TEST(RuntimeCapabilitiesTest, CurrentDeviceOverloadsMatchExplicitDevice)
{
  for (int device = 0; device < rmm::get_num_cuda_devices(); ++device) {
    rmm::cuda_device_id const device_id{device};
    rmm::cuda_set_device_raii set_device{device_id};
    EXPECT_EQ(rmm::detail::runtime_async_alloc::is_supported(),
              rmm::detail::runtime_async_alloc::is_supported(device_id));
    EXPECT_EQ(rmm::detail::concurrent_managed_access::is_supported(),
              rmm::detail::concurrent_managed_access::is_supported(device_id));
    EXPECT_EQ(rmm::detail::runtime_async_managed_alloc::is_supported(),
              rmm::detail::runtime_async_managed_alloc::is_supported(device_id));
    EXPECT_EQ(rmm::detail::device_integrated_memory::is_supported(),
              rmm::detail::device_integrated_memory::is_supported(device_id));
    EXPECT_EQ(rmm::detail::hwdecompress::is_supported(),
              rmm::detail::hwdecompress::is_supported(device_id));
  }
}

TEST(RuntimeCapabilitiesTest, InvalidDeviceIsFalse)
{
  for (auto const device_id :
       {rmm::cuda_device_id{-1}, rmm::cuda_device_id{rmm::get_num_cuda_devices()}}) {
    EXPECT_FALSE(rmm::detail::runtime_async_alloc::is_supported(device_id));
    EXPECT_FALSE(rmm::detail::concurrent_managed_access::is_supported(device_id));
    EXPECT_FALSE(rmm::detail::runtime_async_managed_alloc::is_supported(device_id));
    EXPECT_FALSE(rmm::detail::device_integrated_memory::is_supported(device_id));
    EXPECT_FALSE(rmm::detail::hwdecompress::is_supported(device_id));
  }
}

TEST(PerDeviceCapabilityTest, CachesSuccessfulQueries)
{
  rmm::detail::per_device_capability cache;
  rmm::cuda_device_id const device_id{0};
  int queries{};
  EXPECT_TRUE(cache.get(device_id, [&]() -> std::optional<bool> {
    ++queries;
    return true;
  }));
  EXPECT_TRUE(cache.get(device_id, [&]() -> std::optional<bool> {
    ++queries;
    return false;
  }));
  EXPECT_EQ(queries, 1);
}

TEST(PerDeviceCapabilityTest, DoesNotCacheFailedQueries)
{
  rmm::detail::per_device_capability cache;
  rmm::cuda_device_id const device_id{0};
  EXPECT_FALSE(cache.get(device_id, []() -> std::optional<bool> { return std::nullopt; }));
  EXPECT_TRUE(cache.get(device_id, []() -> std::optional<bool> { return true; }));
}

TEST(PerDeviceCapabilityTest, DoesNotCacheOutOfRangeDevices)
{
  rmm::detail::per_device_capability cache;
  for (auto const device_id :
       {rmm::cuda_device_id{-1}, rmm::cuda_device_id{rmm::get_num_cuda_devices()}}) {
    int queries{};
    auto query = [&]() -> std::optional<bool> {
      ++queries;
      return true;
    };
    EXPECT_TRUE(cache.get(device_id, query));
    EXPECT_TRUE(cache.get(device_id, query));
    EXPECT_EQ(queries, 2);
  }
}

TEST(PerDeviceCapabilityTest, ConcurrentQueriesAgree)
{
  auto const num_devices = rmm::get_num_cuda_devices();
  rmm::detail::per_device_capability cache;
  auto const expected = [](int device) { return device % 2 == 0; };

  constexpr int num_threads{16};
  constexpr int iterations{1000};
  std::atomic<int> mismatches{};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread = 0; thread < num_threads; ++thread) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int device = 0; device < num_devices; ++device) {
          auto const result = cache.get(rmm::cuda_device_id{device},
                                        [&]() -> std::optional<bool> { return expected(device); });
          if (result != expected(device)) { ++mismatches; }
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(mismatches.load(), 0);
}

}  // namespace
}  // namespace rmm::test
