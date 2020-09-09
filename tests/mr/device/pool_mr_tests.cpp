/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include "rmm/detail/aligned.hpp"

#include <gtest/gtest.h>

namespace rmm {
namespace test {
namespace {
using Pool = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
TEST(PoolTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { Pool mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(PoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() { Pool mr{rmm::mr::get_current_device_resource(), 1024, 256}; };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PoolTest, AllocateNinetyPercent)
{
  auto allocate_ninety = []() {
    std::size_t free{}, total{};
    std::tie(free, total) = rmm::mr::detail::available_device_memory();
    auto const ninety_percent_pool =
      rmm::detail::align_up(static_cast<std::size_t>(free * 0.9), Pool::allocation_alignment);
    Pool mr{rmm::mr::get_current_device_resource(), ninety_percent_pool};
  };
  EXPECT_NO_THROW(allocate_ninety());
}

TEST(PoolTest, TwoLargeBuffers)
{
  auto two_large = []() {
    std::size_t free{}, total{};
    std::tie(free, total) = rmm::mr::detail::available_device_memory();
    Pool mr{rmm::mr::get_current_device_resource()};
    auto p1 = mr.allocate(free / 4);
    auto p2 = mr.allocate(free / 4);
    mr.deallocate(p1, free / 4);
    mr.deallocate(p2, free / 4);
  };
  EXPECT_NO_THROW(two_large());
}

TEST(PoolTest, ForceGrowth)
{
  Pool mr{rmm::mr::get_current_device_resource(), 0};
  EXPECT_NO_THROW(mr.allocate(1000));
}

TEST(PoolTest, DeletedStream)
{
  Pool mr{rmm::mr::get_current_device_resource(), 0};
  cudaStream_t stream;
  const int size = 10000;
  cudaStreamCreate(&stream);
  EXPECT_NO_THROW(rmm::device_buffer buff(size, stream, &mr));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_NO_THROW(mr.allocate(size));
}

// Issue #527
TEST(PoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    Pool mr(rmm::mr::get_current_device_resource(), 1000192, 1000192);
    mr.allocate(1000);
  }());
}

TEST(PoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      Pool mr(rmm::mr::get_current_device_resource(), 1000031, 1000192);
      mr.allocate(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      Pool mr(rmm::mr::get_current_device_resource(), 1000192, 1000200);
      mr.allocate(1000);
    }(),
    rmm::logic_error);
}

}  // namespace
}  // namespace test
}  // namespace rmm
