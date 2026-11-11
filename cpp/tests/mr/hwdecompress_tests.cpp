/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace rmm::test {
namespace {

class HWDecompressTest : public ::testing::Test {
 protected:
#if CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  static bool is_decompress_capable(void* ptr)
  {
    bool is_capable{};
    auto const err =
      cuPointerGetAttribute(static_cast<void*>(&is_capable),
                            CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<CUdeviceptr>(ptr));
    EXPECT_EQ(err, CUDA_SUCCESS);
    return is_capable;
  }
#endif

  static void check_decompress_capable(void* ptr)
  {
#if CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
    if (rmm::detail::hwdecompress::is_supported()) {
      EXPECT_TRUE(is_decompress_capable(ptr));
    } else {
      GTEST_SKIP() << "Skipping since hardware decompression is not supported "
                   << "by the current CUDA driver.";
    }
#else
    GTEST_SKIP() << "Skipping since hardware decompression is not supported "
                 << "by the CUDA version used to build RMM.";
#endif
  }
};

TEST_F(HWDecompressTest, CudaMalloc)
{
  const auto allocation_size{100};
  rmm::mr::cuda_memory_resource mr{};
  void* ptr = mr.allocate_sync(allocation_size);
  HWDecompressTest::check_decompress_capable(ptr);
  mr.deallocate_sync(ptr, allocation_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(HWDecompressTest, CudaMallocAsync)
{
  if (!rmm::detail::runtime_async_alloc::is_supported()) {
    GTEST_SKIP() << "Skipping since cudaMallocAsync not supported with this CUDA "
                 << "driver/runtime version";
  }
  const auto pool_init_size{100};
  cudaMemPool_t retained_pool{};
  {
    rmm::mr::cuda_async_memory_resource mr{pool_init_size};
    retained_pool = mr.pool_handle();
    cudaMemPool_t current{};
    RMM_CUDA_TRY(cudaDeviceGetMemPool(&current, rmm::get_current_cuda_device().value()));
    EXPECT_EQ(retained_pool, current);
    std::uint64_t threshold{};
    RMM_CUDA_TRY(
      cudaMemPoolGetAttribute(retained_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
    void* ptr = mr.allocate_sync(pool_init_size);
    HWDecompressTest::check_decompress_capable(ptr);
    mr.deallocate_sync(ptr, pool_init_size);
  }
  cudaMemPool_t current{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&current, rmm::get_current_cuda_device().value()));
  EXPECT_EQ(current, retained_pool);
  void* ptr{};
  RMM_CUDA_TRY(cudaMallocAsync(&ptr, pool_init_size, cudaStream_t{}));
  RMM_CUDA_TRY(cudaFreeAsync(ptr, cudaStream_t{}));
  rmm::mr::cuda_async_memory_resource mr{};
  EXPECT_EQ(mr.pool_handle(), retained_pool);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(HWDecompressTest, ExistingCurrentPoolIsNotModified)
{
#if CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  if (!rmm::detail::hwdecompress::is_supported()) {
    GTEST_SKIP() << "Skipping since hardware decompression is not supported by the current CUDA "
                    "driver.";
  }

  struct temporary_current_pool {
    int device_id{rmm::get_current_cuda_device().value()};
    cudaMemPool_t previous{};
    cudaMemPool_t pool{};
    temporary_current_pool()
    {
      RMM_CUDA_TRY(cudaDeviceGetMemPool(&previous, device_id));
      cudaMemPoolProps props{};
      props.allocType     = cudaMemAllocationTypePinned;
      props.handleTypes   = cudaMemHandleTypeNone;
      props.location.type = cudaMemLocationTypeDevice;
      props.location.id   = device_id;
      RMM_CUDA_TRY(cudaMemPoolCreate(&pool, &props));
      RMM_CUDA_TRY(cudaDeviceSetMemPool(device_id, pool));
    }
    ~temporary_current_pool()
    {
      RMM_ASSERT_CUDA_SUCCESS(cudaDeviceSetMemPool(device_id, previous));
      RMM_ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
    }
    temporary_current_pool(temporary_current_pool const&)            = delete;
    temporary_current_pool& operator=(temporary_current_pool const&) = delete;
  };

  temporary_current_pool current;
  auto const pool = current.pool;
  std::uint64_t expected_threshold{123};
  RMM_CUDA_TRY(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &expected_threshold));
  {
    rmm::mr::cuda_async_memory_resource mr{};
    EXPECT_EQ(mr.pool_handle(), pool);
    void* ptr = mr.allocate_sync(100);
    EXPECT_FALSE(HWDecompressTest::is_decompress_capable(ptr));
    mr.deallocate_sync(ptr, 100);
  }
  std::uint64_t threshold{};
  RMM_CUDA_TRY(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, expected_threshold);
#else
  GTEST_SKIP() << "Skipping since hardware decompression is not supported by the CUDA version "
                  "used to build RMM.";
#endif
}

}  // namespace
}  // namespace rmm::test
