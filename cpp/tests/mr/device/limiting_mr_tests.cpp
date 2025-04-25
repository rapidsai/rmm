/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "../../byte_literals.hpp"

#include <rmm/error.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using limiting_adaptor = rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>;

TEST(LimitingTest, ThrowOnNullUpstream)
{
  auto const max_size{5_MiB};
  auto construct_nullptr = []() { limiting_adaptor mr{nullptr, max_size}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(LimitingTest, TooBig)
{
  auto const max_size{5_MiB};
  limiting_adaptor mr{rmm::mr::get_current_device_resource_ref(), max_size};
  EXPECT_THROW(mr.allocate(max_size + 1), rmm::out_of_memory);
}

TEST(LimitingTest, UpstreamFailure)
{
  auto const max_size_1{2_MiB};
  auto const max_size_2{5_MiB};
  limiting_adaptor mr1{rmm::mr::get_current_device_resource_ref(), max_size_1};
  limiting_adaptor mr2{&mr1, max_size_2};
  EXPECT_THROW(mr2.allocate(4_MiB), rmm::out_of_memory);
}

TEST(LimitingTest, UnderLimitDueToFrees)
{
  auto const max_size{10_MiB};
  limiting_adaptor mr{rmm::mr::get_current_device_resource_ref(), max_size};
  auto const size1{4_MiB};
  auto* ptr1           = mr.allocate(size1);
  auto allocated_bytes = size1;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  auto* ptr2 = mr.allocate(size1);
  allocated_bytes += size1;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  mr.deallocate(ptr1, size1);
  allocated_bytes -= size1;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  // note that we don't keep track of fragmentation or things like page size
  // so this should fill 100% of the memory even though it is probably over.
  auto const size2{6_MiB};
  auto* ptr3 = mr.allocate(size2);
  allocated_bytes += size2;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), 0);
  mr.deallocate(ptr2, size1);
  mr.deallocate(ptr3, size2);
}

TEST(LimitingTest, OverLimit)
{
  auto const max_size{10_MiB};
  limiting_adaptor mr{rmm::mr::get_current_device_resource_ref(), max_size};
  auto const size1{4_MiB};
  auto* ptr1           = mr.allocate(size1);
  auto allocated_bytes = size1;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  auto* ptr2 = mr.allocate(size1);
  allocated_bytes += size1;
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  auto const size2{3_MiB};
  EXPECT_THROW(mr.allocate(size2), rmm::out_of_memory);
  EXPECT_EQ(mr.get_allocated_bytes(), allocated_bytes);
  EXPECT_EQ(mr.get_allocation_limit() - mr.get_allocated_bytes(), max_size - allocated_bytes);
  mr.deallocate(ptr1, 4_MiB);
  mr.deallocate(ptr2, 4_MiB);
}

}  // namespace
}  // namespace rmm::test
