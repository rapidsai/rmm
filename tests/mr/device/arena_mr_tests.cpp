/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {
using arena_mr = rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource>;

TEST(ArenaTest, NullUpstream)
{
  EXPECT_THROW([]() { arena_mr mr{nullptr}; }(), rmm::logic_error);
}

TEST(ArenaTest, AllocateNinetyPercent)
{
  EXPECT_NO_THROW([]() {
    auto const free = rmm::detail::available_device_memory().first;
    auto const ninety_percent =
      rmm::detail::align_up_cuda(static_cast<std::size_t>(static_cast<double>(free) * 0.9));
    arena_mr mr(rmm::mr::get_current_device_resource(), ninety_percent);
  }());
}

TEST(ArenaTest, SmallMediumLarge)
{
  EXPECT_NO_THROW([]() {
    arena_mr mr(rmm::mr::get_current_device_resource());
    auto* small = mr.allocate(256);
    auto* medium = mr.allocate(1U << 26U);
    auto const free = rmm::detail::available_device_memory().first;
    auto* large = mr.allocate(free / 2);
    mr.deallocate(small, 256);
    mr.deallocate(medium, 1U << 26U);
    mr.deallocate(large, free / 4);
  }());
}

}  // namespace
}  // namespace rmm::test
