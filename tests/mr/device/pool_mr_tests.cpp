/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;

namespace rmm::test {
namespace {
using cuda_mr     = rmm::mr::cuda_memory_resource;
using pool_mr     = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
using limiting_mr = rmm::mr::limiting_resource_adaptor<rmm::mr::cuda_memory_resource>;

TEST(PoolTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { pool_mr mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(PoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{rmm::mr::get_current_device_resource(), initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PoolTest, AllocateNinetyPercent)
{
  auto allocate_ninety = []() {
    auto const [free, total] = rmm::detail::available_device_memory();
    (void)total;
    auto const ninety_percent_pool =
      rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.9),
                            rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    pool_mr mr{rmm::mr::get_current_device_resource(), ninety_percent_pool};
  };
  EXPECT_NO_THROW(allocate_ninety());
}

TEST(PoolTest, TwoLargeBuffers)
{
  auto two_large = []() {
    auto const [free, total] = rmm::detail::available_device_memory();
    (void)total;
    pool_mr mr{rmm::mr::get_current_device_resource()};
    auto* ptr1 = mr.allocate(free / 4);
    auto* ptr2 = mr.allocate(free / 4);
    mr.deallocate(ptr1, free / 4);
    mr.deallocate(ptr2, free / 4);
  };
  EXPECT_NO_THROW(two_large());
}

TEST(PoolTest, ForceGrowth)
{
  cuda_mr cuda;
  {
    auto const max_size{6000};
    limiting_mr limiter{&cuda, max_size};
    pool_mr mr{&limiter, 0};
    EXPECT_NO_THROW(mr.allocate(1000));
    EXPECT_NO_THROW(mr.allocate(4000));
    EXPECT_NO_THROW(mr.allocate(500));
    EXPECT_THROW(mr.allocate(2000), rmm::out_of_memory);  // too much
  }
  {
    // with max pool size
    auto const max_size{6000};
    limiting_mr limiter{&cuda, max_size};
    pool_mr mr{&limiter, 0, 8192};
    EXPECT_NO_THROW(mr.allocate(1000));
    EXPECT_THROW(mr.allocate(4000), rmm::out_of_memory);  // too much
    EXPECT_NO_THROW(mr.allocate(500));
    EXPECT_NO_THROW(mr.allocate(2000));  // fits
  }
}

TEST(PoolTest, DeletedStream)
{
  pool_mr mr{rmm::mr::get_current_device_resource(), 0};
  cudaStream_t stream{};  // we don't use rmm::cuda_stream here to make destruction more explicit
  const int size = 10000;
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  EXPECT_NO_THROW(rmm::device_buffer buff(size, cuda_stream_view{stream}, &mr));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_NO_THROW(mr.allocate(size));
}

// Issue #527
TEST(PoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::get_current_device_resource(), 1000192, 1000192);
    mr.allocate(1000);
  }());
}

TEST(PoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource(), 1000031, 1000192);
      mr.allocate(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource(), 1000192, 1000200);
      mr.allocate(1000);
    }(),
    rmm::logic_error);
}

TEST(PoolTest, UpstreamDoesntSupportMemInfo)
{
  cuda_mr cuda;
  pool_mr mr1(&cuda);
  pool_mr mr2(&mr1);
  auto* ptr = mr2.allocate(1024);
  mr2.deallocate(ptr, 1024);
}

TEST(PoolTest, MultidevicePool)
{
  using MemoryResource = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;

  // Get the number of cuda devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::mr::cuda_memory_resource general_mr;

    // initializing pool_memory_resource of multiple devices
    int devices      = 2;
    size_t pool_size = 1024;
    std::vector<std::shared_ptr<MemoryResource>> mrs;

    for (int i = 0; i < devices; ++i) {
      RMM_CUDA_TRY(cudaSetDevice(i));
      auto mr = std::make_shared<MemoryResource>(&general_mr, pool_size, pool_size);
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, mr.get());
      mrs.emplace_back(mr);
    }

    {
      RMM_CUDA_TRY(cudaSetDevice(0));
      rmm::device_buffer buf_a(16, rmm::cuda_stream_per_thread, mrs[0].get());

      {
        RMM_CUDA_TRY(cudaSetDevice(1));
        rmm::device_buffer buf_b(16, rmm::cuda_stream_per_thread, mrs[1].get());
      }

      RMM_CUDA_TRY(cudaSetDevice(0));
    }
  }
}

}  // namespace
}  // namespace rmm::test
