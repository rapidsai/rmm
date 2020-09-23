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
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

#include <gtest/gtest.h>
#include "mr_test.hpp"

namespace rmm {
namespace test {
namespace {
using Limiting_adaptor = rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>;
TEST(LimitingTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { Limiting_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(LimitingTest, TooBig)
{
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 1_MiB};
  EXPECT_THROW(mr.allocate(5_MiB), rmm::bad_alloc);
}

TEST(LimitingTest, UnderLimitDueToFrees)
{
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 10_MiB};
  auto p1 = mr.allocate(4_MiB);
  EXPECT_EQ(mr.space_free(), 6_MiB);
  auto p2 = mr.allocate(4_MiB);
  EXPECT_EQ(mr.space_free(), 2_MiB);
  mr.deallocate(p1, 4_MiB);
  EXPECT_EQ(mr.space_free(), 6_MiB);
  // note that we don't keep track of fragmentation, so this should fill
  // 100% of the memory even though it is probably over.
  EXPECT_NO_THROW(mr.allocate(6_MiB));
  EXPECT_EQ(mr.space_free(), 0);
}

TEST(LimitingTest, OverLimit)
{
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 10_MiB};
  auto p1 = mr.allocate(4_MiB);
  EXPECT_EQ(mr.space_free(), 6_MiB);
  auto p2 = mr.allocate(4_MiB);
  EXPECT_EQ(mr.space_free(), 2_MiB);
  EXPECT_THROW(mr.allocate(3_MiB), rmm::bad_alloc);
  EXPECT_EQ(mr.space_free(), 2_MiB);
}

}  // namespace
}  // namespace test
}  // namespace rmm
