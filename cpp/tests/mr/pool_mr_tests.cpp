/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace rmm::test {
namespace {
using cuda_mr     = rmm::mr::cuda_memory_resource;
using pool_mr     = rmm::mr::pool_memory_resource;
using limiting_mr = rmm::mr::limiting_resource_adaptor<rmm::mr::cuda_memory_resource>;

TEST(PoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PoolTest, AllocateNinetyPercent)
{
  auto allocate_ninety = []() {
    auto const [free, total] = rmm::available_device_memory();
    (void)total;
    auto const ninety_percent_pool = rmm::percent_of_free_device_memory(90);
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), ninety_percent_pool};
  };
  EXPECT_NO_THROW(allocate_ninety());
}

TEST(PoolTest, TwoLargeBuffers)
{
  auto two_large = []() {
    [[maybe_unused]] auto const [free, total] = rmm::available_device_memory();
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), rmm::percent_of_free_device_memory(50)};
    auto* ptr1 = mr.allocate_sync(free / 4);
    auto* ptr2 = mr.allocate_sync(free / 4);
    mr.deallocate_sync(ptr1, free / 4);
    mr.deallocate_sync(ptr2, free / 4);
  };
  EXPECT_NO_THROW(two_large());
}

TEST(PoolTest, ForceGrowth)
{
  cuda_mr cuda;
  {
    auto const max_size{6000};
    limiting_mr limiter{&cuda, max_size};
    pool_mr mr{limiter, 0};
    EXPECT_NO_THROW(mr.allocate_sync(1000));
    EXPECT_NO_THROW(mr.allocate_sync(4000));
    EXPECT_NO_THROW(mr.allocate_sync(500));
    EXPECT_THROW(mr.allocate_sync(2000), rmm::out_of_memory);  // too much
  }
  {
    // with max pool size
    auto const max_size{6000};
    limiting_mr limiter{&cuda, max_size};
    pool_mr mr{limiter, 0, 8192};
    EXPECT_NO_THROW(mr.allocate_sync(1000));
    EXPECT_THROW(mr.allocate_sync(4000), rmm::out_of_memory);  // too much
    EXPECT_NO_THROW(mr.allocate_sync(500));
    EXPECT_NO_THROW(mr.allocate_sync(2000));  // fits
  }
}

TEST(PoolTest, DeletedStream)
{
  pool_mr mr{rmm::mr::get_current_device_resource_ref(), 0};
  cudaStream_t stream{};  // we don't use rmm::cuda_stream here to make destruction more explicit
  const int size = 10000;
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  EXPECT_NO_THROW(rmm::device_buffer buff(size, cuda_stream_view{stream}, &mr));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_NO_THROW(mr.allocate_sync(size));
}

// Issue #527
TEST(PoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000192, 1000192);
    mr.allocate_sync(1000);
  }());
}

TEST(PoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000031, 1000192);
      mr.allocate_sync(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000192, 1000200);
      mr.allocate_sync(1000);
    }(),
    rmm::logic_error);
}

TEST(PoolTest, UpstreamDoesntSupportMemInfo)
{
  cuda_mr cuda;
  pool_mr mr1{cuda, 0};
  pool_mr mr2{mr1, 0};
  auto* ptr = mr2.allocate_sync(1024);
  mr2.deallocate_sync(ptr, 1024);
}

TEST(PoolTest, MultidevicePool)
{
  // Get the number of CUDA devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::mr::cuda_memory_resource general_mr;

    // initializing pool_memory_resource of multiple devices
    int devices      = 2;
    size_t pool_size = 1024;
    std::vector<pool_mr> mrs;

    for (int i = 0; i < devices; ++i) {
      RMM_CUDA_TRY(cudaSetDevice(i));
      auto mr = pool_mr{general_mr, pool_size, pool_size};
      rmm::mr::set_per_device_resource_ref(rmm::cuda_device_id{i}, mr);
      mrs.emplace_back(mr);
    }

    {
      RMM_CUDA_TRY(cudaSetDevice(0));
      rmm::device_buffer buf_a(16, rmm::cuda_stream_per_thread, mrs[0]);

      {
        RMM_CUDA_TRY(cudaSetDevice(1));
        rmm::device_buffer buf_b(16, rmm::cuda_stream_per_thread, mrs[1]);
      }

      RMM_CUDA_TRY(cudaSetDevice(0));
    }
  }
}

class PoolMemoryResourceTest : public ::testing::Test {
 protected:
  rmm::mr::pool_memory_resource pool{rmm::mr::get_current_device_resource_ref(), 1024 * 1024};
};

TEST_F(PoolMemoryResourceTest, GetUpstreamResource)
{
  [[maybe_unused]] auto ref = pool.get_upstream_resource();
}

TEST_F(PoolMemoryResourceTest, AllocateDeallocate)
{
  constexpr std::size_t size{4096};
  auto* ptr = pool.allocate_sync(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(pool.deallocate_sync(ptr, size));
}

TEST_F(PoolMemoryResourceTest, SharedOwnership)
{
  auto copy = pool;  // copy shares the same underlying state
  constexpr std::size_t size{4096};
  auto* ptr = pool.allocate_sync(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, size));  // deallocate through the copy
}

TEST_F(PoolMemoryResourceTest, Equality)
{
  auto copy = pool;
  EXPECT_EQ(pool, copy);

  rmm::mr::pool_memory_resource other{rmm::mr::get_current_device_resource_ref(), 1024 * 1024};
  EXPECT_NE(pool, other);
}

TEST_F(PoolMemoryResourceTest, PoolSize) { EXPECT_GE(pool.pool_size(), 1024 * 1024); }

}  // namespace

namespace test_properties {

// pool_memory_resource is now non-template; verify it satisfies the resource concept
// and always has device_accessible (since upstream is any_resource<device_accessible>)
static_assert(cuda::mr::resource_with<rmm::mr::pool_memory_resource, cuda::mr::device_accessible>);

}  // namespace test_properties

}  // namespace rmm::test
