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

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using tracking_adaptor      = rmm::mr::tracking_resource_adaptor<rmm::mr::device_memory_resource>;
using aligned_adaptor       = rmm::mr::aligned_resource_adaptor<tracking_adaptor>;

TEST(AlignedTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { aligned_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(AlignedTest, SmallAllocations)
{
  tracking_adaptor tracker{rmm::mr::get_current_device_resource()};
  aligned_adaptor mr{&tracker};

  std::vector<void *> allocations;
  allocations.reserve(16);
  for (int i = 0; i < 16; ++i) {
    allocations.push_back(mr.allocate((i + 1) * 256));
  }

  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 16);
  EXPECT_EQ(tracker.get_allocated_bytes(), 256 * 136);

  for (int i = 0; i < 16; ++i) {
    mr.deallocate(allocations[i], (i + 1) * 256);
  }
  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(tracker.get_allocated_bytes(), 0);
}

TEST(AlignedTest, LargeAllocations)
{
  tracking_adaptor tracker{rmm::mr::get_current_device_resource()};
  aligned_adaptor mr{&tracker};

  std::vector<void *> allocations;
  allocations.reserve(3);
  for (int i = 0; i < 3; ++i) {
    allocations.push_back(mr.allocate(4096 + (i + 1) * 256));
  }

  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 3);
  EXPECT_EQ(tracker.get_allocated_bytes(), 4096 * 6);

  for (int i = 0; i < 3; ++i) {
    mr.deallocate(allocations[i], 4096 + (i + 1) * 256);
  }
  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(tracker.get_allocated_bytes(), 0);
}

TEST(AlignedTest, SmallAllocationsWithCustomAlignmentSize)
{
  tracking_adaptor tracker{rmm::mr::get_current_device_resource()};
  aligned_adaptor mr{&tracker, {8192}};

  std::vector<void *> allocations;
  allocations.reserve(32);
  for (int i = 0; i < 32; ++i) {
    allocations.push_back(mr.allocate((i + 1) * 256));
  }

  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 32);
  EXPECT_EQ(tracker.get_allocated_bytes(), 256 * 528);

  for (int i = 0; i < 32; ++i) {
    mr.deallocate(allocations[i], (i + 1) * 256);
  }
  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(tracker.get_allocated_bytes(), 0);
}

TEST(AlignedTest, LargeAllocationsWithCustomAlignmentSize)
{
  tracking_adaptor tracker{rmm::mr::get_current_device_resource()};
  aligned_adaptor mr{&tracker, {8192}};

  std::vector<void *> allocations;
  allocations.reserve(7);
  for (int i = 0; i < 7; ++i) {
    allocations.push_back(mr.allocate(8192 + (i + 1) * 256));
  }

  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 7);
  EXPECT_EQ(tracker.get_allocated_bytes(), 8192 * 14);

  for (int i = 0; i < 7; ++i) {
    mr.deallocate(allocations[i], 8192 + (i + 1) * 256);
  }
  EXPECT_EQ(tracker.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(tracker.get_allocated_bytes(), 0);
}

}  // namespace
}  // namespace rmm::test
