/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

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

class temporary_current_pool {
 public:
  explicit temporary_current_pool(std::uint64_t release_threshold)
    : device_id_{rmm::get_current_cuda_device().value()}
  {
    RMM_CUDA_TRY(cudaDeviceGetMemPool(&previous_, device_id_));
    cudaMemPoolProps props{};
    props.allocType     = cudaMemAllocationTypePinned;
    props.handleTypes   = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id   = device_id_;
    RMM_CUDA_TRY(cudaMemPoolCreate(&pool_, &props));
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &release_threshold));
    RMM_CUDA_TRY(cudaDeviceSetMemPool(device_id_, pool_));
  }

  ~temporary_current_pool()
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaDeviceSetMemPool(device_id_, previous_));
    RMM_ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool_));
  }

  temporary_current_pool(temporary_current_pool const&)            = delete;
  temporary_current_pool& operator=(temporary_current_pool const&) = delete;

  [[nodiscard]] cudaMemPool_t get() const noexcept { return pool_; }

 private:
  int device_id_{};
  cudaMemPool_t previous_{};
  cudaMemPool_t pool_{};
};

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

TEST_F(AsyncMRTest, DefaultResourcesShareCurrentPool)
{
  cuda_async_mr mr1{};
  cuda_async_mr mr2{};
  cudaMemPool_t current{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&current, rmm::get_current_cuda_device().value()));
  EXPECT_EQ(mr1.pool_handle(), current);
  EXPECT_EQ(mr2.pool_handle(), current);
  EXPECT_EQ(mr1, mr2);
}

TEST_F(AsyncMRTest, DefaultPoolReleaseThresholdIsUint64Max)
{
  if (rmm::detail::hwdecompress::is_supported()) {
    GTEST_SKIP() << "Skipping since hardware decompression replaces the device-default pool.";
  }
  auto const device_id = rmm::get_current_cuda_device().value();
  cudaMemPool_t default_pool{};
  RMM_CUDA_TRY(cudaDeviceGetDefaultMemPool(&default_pool, device_id));
  std::uint64_t threshold{123};
  RMM_CUDA_TRY(cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));

  cuda_async_mr mr{};
  EXPECT_EQ(mr.pool_handle(), default_pool);
  RMM_CUDA_TRY(cudaMemPoolGetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
}

TEST_F(AsyncMRTest, ExistingCurrentPoolIsNotModified)
{
  std::uint64_t const expected_threshold{123};
  temporary_current_pool current{expected_threshold};
  {
    cuda_async_mr mr{};
    EXPECT_EQ(mr.pool_handle(), current.get());
  }

  cudaMemPool_t selected{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&selected, rmm::get_current_cuda_device().value()));
  EXPECT_EQ(selected, current.get());
  std::uint64_t threshold{};
  RMM_CUDA_TRY(cudaMemPoolGetAttribute(current.get(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, expected_threshold);
}

TEST_F(AsyncMRTest, ExplicitReleaseThresholdIsApplied)
{
  const std::uint64_t pool_release_threshold{1000};
  cudaMemPool_t current{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&current, rmm::get_current_cuda_device().value()));
  cuda_async_mr mr{{}, pool_release_threshold};
  std::uint64_t threshold{0};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, pool_release_threshold);
  EXPECT_NE(mr.pool_handle(), current);
  cudaMemPool_t selected{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&selected, rmm::get_current_cuda_device().value()));
  EXPECT_EQ(selected, current);
}

TEST_F(AsyncMRTest, DirectAllocationsShareCurrentPool)
{
  cuda_async_mr mr{};
  cudaStream_t first{};
  cudaStream_t second{};
  RMM_CUDA_TRY(cudaStreamCreate(&first));
  RMM_CUDA_TRY(cudaStreamCreate(&second));
  constexpr std::size_t size{1024};

  void* direct{};
  RMM_CUDA_TRY(cudaMallocAsync(&direct, size, first));
  RMM_CUDA_TRY(cudaStreamSynchronize(first));
  mr.deallocate(cuda::stream_ref{second}, direct, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  void* from_rmm = mr.allocate(cuda::stream_ref{first}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaStreamSynchronize(first));
  RMM_CUDA_TRY(cudaFreeAsync(from_rmm, second));

  RMM_CUDA_TRY(cudaStreamSynchronize(second));
  RMM_CUDA_TRY(cudaStreamDestroy(first));
  RMM_CUDA_TRY(cudaStreamDestroy(second));
}

TEST_F(AsyncMRTest, ZeroReleaseThresholdIsUint64Max)
{
  cuda_async_mr mr{{}, 0};
  std::uint64_t threshold{};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
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

TEST_F(AsyncMRTest, ExportHandleUsesOwnedPool)
{
  auto const handle_type =
    rmm::mr::cuda_async_memory_resource::allocation_handle_type::posix_file_descriptor;
  if (!rmm::detail::export_handle_type::is_supported(
        static_cast<cudaMemAllocationHandleType>(handle_type))) {
    GTEST_SKIP() << "POSIX file descriptor handles are not supported in this environment.";
  }
  cudaMemPool_t current{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&current, rmm::get_current_cuda_device().value()));
  cuda_async_mr mr{{}, {}, handle_type};
  EXPECT_NE(mr.pool_handle(), current);
  std::uint64_t threshold{};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));
  EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
  cudaMemPool_t selected{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&selected, rmm::get_current_cuda_device().value()));
  EXPECT_EQ(selected, current);
}

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
