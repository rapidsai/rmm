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

#define MB << 20

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
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 1 MB};
  EXPECT_THROW(mr.allocate(5 MB), rmm::cuda_error);
}

TEST(LimitingTest, UnderLimitDueToFrees)
{
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 10 MB};
  auto p1 = mr.allocate(4 MB);
  EXPECT_EQ(mr.space_free(), 6 MB);
  auto p2 = mr.allocate(4 MB);
  EXPECT_EQ(mr.space_free(), 2 MB);
  mr.deallocate(p1, 4 MB);
  EXPECT_EQ(mr.space_free(), 6 MB);
  // note that we don't keep track of fragmentation, so this should fill
  // 100% of the memory even though it is probably over.
  EXPECT_NO_THROW(mr.allocate(6 MB));
  EXPECT_EQ(mr.space_free(), 0);
}

TEST(LimitingTest, OverLimit)
{
  Limiting_adaptor mr{rmm::mr::get_current_device_resource(), 10 MB};
  auto p1 = mr.allocate(4 MB);
  EXPECT_EQ(mr.space_free(), 6 MB);
  auto p2 = mr.allocate(4 MB);
  EXPECT_EQ(mr.space_free(), 2 MB);
  EXPECT_THROW(mr.allocate(3 MB), rmm::cuda_error);
  EXPECT_EQ(mr.space_free(), 2 MB);
}

}  // namespace
}  // namespace test
}  // namespace rmm
