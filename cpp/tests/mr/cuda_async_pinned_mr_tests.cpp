/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_pinned_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <optional>
#include <vector>

namespace rmm::test {
namespace {

using cuda_async_pinned_mr = rmm::mr::cuda_async_pinned_memory_resource;

static_assert(cuda::mr::synchronous_resource_with<cuda_async_pinned_mr, cuda::mr::host_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<cuda_async_pinned_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_async_pinned_mr, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<cuda_async_pinned_mr, cuda::mr::device_accessible>);

class AsyncPinnedMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::runtime_async_pinned_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests because stream-ordered pinned host allocation is "
                      "unsupported by this CUDA driver/runtime.";
    }
  }
};

struct host_callback_gate {
  std::mutex mutex;
  std::condition_variable condition;
  bool started{};
  bool released{};
};

void release(host_callback_gate& gate) noexcept
{
  {
    std::lock_guard lock{gate.mutex};
    gate.released = true;
  }
  gate.condition.notify_all();
}

struct callback_guard {
  host_callback_gate& gate;
  cudaStream_t stream;

  ~callback_guard()
  {
    release(gate);
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream));
  }
};

TEST_F(AsyncPinnedMRTest, BasicAllocateDeallocate)
{
  constexpr std::size_t size{1_KiB};
  cuda_async_pinned_mr mr{};

  auto* ptr = mr.allocate_sync(size);
  ASSERT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, size);

  EXPECT_EQ(mr.allocate_sync(0), nullptr);
  EXPECT_NO_THROW(mr.deallocate_sync(nullptr, 0));

  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  EXPECT_EQ(mr.allocate(stream, 0, rmm::CUDA_ALLOCATION_ALIGNMENT), nullptr);
  EXPECT_NO_THROW(mr.deallocate(stream, nullptr, 0, rmm::CUDA_ALLOCATION_ALIGNMENT));
}

TEST_F(AsyncPinnedMRTest, AllocatedPointerIsPinnedAndHostAccessible)
{
  constexpr std::size_t size{4_KiB};
  constexpr std::uint8_t expected_value{0x2a};
  cuda_async_pinned_mr mr{};

  auto* ptr = static_cast<std::uint8_t*>(mr.allocate_sync(size));
  ASSERT_NE(ptr, nullptr);

  cudaPointerAttributes attributes{};
  RMM_CUDA_TRY(cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  EXPECT_EQ(attributes.hostPointer, ptr);
  EXPECT_NE(attributes.devicePointer, nullptr);

  std::fill_n(ptr, size, expected_value);
  EXPECT_TRUE(
    std::all_of(ptr, ptr + size, [](std::uint8_t value) { return value == expected_value; }));

  mr.deallocate_sync(ptr, size);
}

TEST_F(AsyncPinnedMRTest, SupportedAlignments)
{
  cuda_async_pinned_mr mr{};

  for (std::size_t alignment = 1; alignment <= rmm::CUDA_ALLOCATION_ALIGNMENT; alignment *= 2) {
    constexpr std::size_t size{513};
    auto* ptr = mr.allocate_sync(size, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(rmm::is_pointer_aligned(ptr, alignment)) << "alignment: " << alignment;
    mr.deallocate_sync(ptr, size, alignment);
  }
}

TEST_F(AsyncPinnedMRTest, RejectsUnsupportedAlignment)
{
  cuda_async_pinned_mr mr{};
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

  for (auto alignment : std::array<std::size_t, 3>{0, 192, 512}) {
    EXPECT_THROW((void)mr.allocate_sync(100, alignment), rmm::bad_alloc)
      << "alignment: " << alignment;
    EXPECT_THROW((void)mr.allocate(stream, 100, alignment), rmm::bad_alloc)
      << "alignment: " << alignment;
  }
}

TEST_F(AsyncPinnedMRTest, AllInstancesSharePoolAndCompareEqual)
{
  cuda_async_pinned_mr first{};
  cuda_async_pinned_mr second{};
  auto copy = first;

  ASSERT_NE(first.pool_handle(), nullptr);
  EXPECT_EQ(first.pool_handle(), second.pool_handle());
  EXPECT_EQ(first.pool_handle(), copy.pool_handle());
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, copy);

  constexpr std::size_t size{1_KiB};
  auto* ptr = first.allocate_sync(size);
  EXPECT_NO_THROW(second.deallocate_sync(ptr, size));
}

TEST_F(AsyncPinnedMRTest, CopyKeepsResourceAlive)
{
  std::optional<cuda_async_pinned_mr> copy;
  {
    cuda_async_pinned_mr original{};
    copy.emplace(original);
    EXPECT_EQ(original, *copy);
  }

  constexpr std::size_t size{1_KiB};
  auto* ptr = copy->allocate_sync(size);
  ASSERT_NE(ptr, nullptr);
  copy->deallocate_sync(ptr, size);
}

TEST_F(AsyncPinnedMRTest, PoolHasDefaultReleaseThreshold)
{
  cuda_async_pinned_mr mr{};
  std::uint64_t release_threshold{1};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrReleaseThreshold, &release_threshold));
  EXPECT_EQ(release_threshold, std::uint64_t{0});
}

TEST_F(AsyncPinnedMRTest, AllocationsUseExposedPool)
{
  constexpr std::size_t size{1_MiB};
  cuda_async_pinned_mr mr{};

  std::uint64_t used_before{};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used_before));

  auto* ptr = mr.allocate_sync(size);
  ASSERT_NE(ptr, nullptr);

  std::uint64_t used_during{};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used_during));
  EXPECT_GE(used_during, used_before + size);

  mr.deallocate_sync(ptr, size);

  std::uint64_t used_after{};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used_after));
  EXPECT_EQ(used_after, used_before);
}

TEST_F(AsyncPinnedMRTest, PoolIsAccessibleFromAllVisibleDevices)
{
  cuda_async_pinned_mr mr{};
  int device_count{};
  RMM_CUDA_TRY(cudaGetDeviceCount(&device_count));

  for (int device = 0; device < device_count; ++device) {
    cudaMemLocation location{.type = cudaMemLocationTypeDevice, .id = device};
    cudaMemAccessFlags flags{cudaMemAccessFlagsProtNone};
    RMM_CUDA_TRY(cudaMemPoolGetAccess(&flags, mr.pool_handle(), &location));
    EXPECT_EQ(flags, cudaMemAccessFlagsProtReadWrite) << "device: " << device;
  }
}

#if CUDART_VERSION >= 13000
TEST_F(AsyncPinnedMRTest, UsesCudaDefaultPinnedHostPool)
{
  cuda_async_pinned_mr mr{};
  cudaMemLocation location{.type = cudaMemLocationTypeHost, .id = 0};
  cudaMemPool_t default_pool{};
  RMM_CUDA_TRY(cudaMemGetDefaultMemPool(&default_pool, &location, cudaMemAllocationTypePinned));
  EXPECT_EQ(mr.pool_handle(), default_pool);

  cudaMemAllocationType allocation_type{cudaMemAllocationTypeInvalid};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrAllocationType, &allocation_type));
  EXPECT_EQ(allocation_type, cudaMemAllocationTypePinned);

  cudaMemLocationType location_type{cudaMemLocationTypeNone};
  RMM_CUDA_TRY(
    cudaMemPoolGetAttribute(mr.pool_handle(), cudaMemPoolAttrLocationType, &location_type));
  EXPECT_EQ(location_type, cudaMemLocationTypeHost);
}
#endif

TEST_F(AsyncPinnedMRTest, DeallocationIsStreamOrdered)
{
  constexpr std::size_t size{1_MiB};
  constexpr std::uint8_t expected_value{0x2a};
  constexpr auto callback_timeout     = std::chrono::seconds{10};
  constexpr auto deallocation_timeout = std::chrono::seconds{5};

  cuda_async_pinned_mr mr{};
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{size, stream};

  auto* pinned = static_cast<std::uint8_t*>(mr.allocate_sync(size));
  ASSERT_NE(pinned, nullptr);
  std::fill_n(pinned, size, expected_value);

  host_callback_gate gate;
  std::future<void> deallocation;
  callback_guard guard{gate, stream.value()};

  RMM_CUDA_TRY(cudaLaunchHostFunc(
    stream.value(),
    [](void* data) {
      auto& gate = *static_cast<host_callback_gate*>(data);
      std::unique_lock lock{gate.mutex};
      gate.started = true;
      gate.condition.notify_all();
      gate.condition.wait(lock, [&gate] { return gate.released; });
    },
    &gate));

  bool callback_started{};
  {
    std::unique_lock lock{gate.mutex};
    callback_started =
      gate.condition.wait_for(lock, callback_timeout, [&gate] { return gate.started; });
  }
  if (!callback_started) {
    release(gate);
    stream.synchronize();
    mr.deallocate_sync(pinned, size);
    FAIL() << "CUDA host callback did not start before timeout";
  }

  RMM_CUDA_TRY(
    cudaMemcpyAsync(device.data(), pinned, size, cudaMemcpyHostToDevice, stream.value()));

  int device_id{};
  RMM_CUDA_TRY(cudaGetDevice(&device_id));
  deallocation                   = std::async(std::launch::async, [&, device_id] {
    RMM_CUDA_TRY(cudaSetDevice(device_id));
    mr.deallocate(stream, pinned, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  });
  auto const deallocation_status = deallocation.wait_for(deallocation_timeout);

  // Always release the callback before waiting on the asynchronous task. With cudaFreeHost, the
  // task remains blocked until the preceding copy completes; cudaFreeAsync returns immediately
  // after ordering the free behind the copy.
  release(gate);
  deallocation.get();
  stream.synchronize();

  std::vector<std::uint8_t> result(size);
  RMM_CUDA_TRY(cudaMemcpy(result.data(), device.data(), size, cudaMemcpyDeviceToHost));

  EXPECT_EQ(deallocation_status, std::future_status::ready);
  EXPECT_TRUE(std::all_of(
    result.begin(), result.end(), [](std::uint8_t value) { return value == expected_value; }));
}

}  // namespace
}  // namespace rmm::test
