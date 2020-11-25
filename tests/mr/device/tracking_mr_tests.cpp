/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include "mr_test.hpp"

#include <gtest/gtest.h>

namespace rmm {
namespace test {
namespace {

using tracking_adaptor = rmm::mr::tracking_resource_adaptor<rmm::mr::device_memory_resource>;

TEST(TrackingTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { tracking_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(TrackingTest, Empty)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, AllFreed)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  for (auto p : allocations) {
    mr.deallocate(p, 10_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, AllocationsLeftWithStacks)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource(), true};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  for (int i = 0; i < 10; i += 2) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 5);
  EXPECT_EQ(mr.get_allocated_bytes(), 50_MiB);
  auto const &outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), 5);
  EXPECT_NE(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, AllocationsLeftWithoutStacks)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  for (int i = 0; i < 10; i += 2) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 5);
  EXPECT_EQ(mr.get_allocated_bytes(), 50_MiB);
  auto const &outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), 5);
  EXPECT_EQ(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, PeakAllocations)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  // Delete every other allocation
  for (auto &&it = allocations.begin(); it != allocations.end(); ++it) {
    mr.deallocate(*it, 10_MiB);
    it = allocations.erase(it);
  }

  auto current_alloc_counts = mr.get_total_allocation_counts();

  // Verify current allocations
  EXPECT_EQ(mr.get_allocated_bytes(), 50_MiB);
  EXPECT_EQ(current_alloc_counts.current_bytes, 50_MiB);
  EXPECT_EQ(current_alloc_counts.current_count, 5);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_counts.peak_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 10);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 10);

  // Add 10 more to increase the peak
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  // Deallocate all remaining
  for (int i = 0; i < allocations.size(); ++i) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  allocations.clear();

  current_alloc_counts = mr.get_total_allocation_counts();

  // Verify current allocations
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(current_alloc_counts.current_bytes, 0);
  EXPECT_EQ(current_alloc_counts.current_count, 0);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_counts.peak_bytes, 150_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 15);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 200_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 20);
}

TEST(TrackingTest, MultiTracking)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  rmm::mr::set_current_device_resource(&mr);

  std::vector<std::shared_ptr<rmm::device_buffer>> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.emplace_back(std::make_shared<rmm::device_buffer>(10_MiB));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 10);

  tracking_adaptor inner_mr{rmm::mr::get_current_device_resource()};
  rmm::mr::set_current_device_resource(&inner_mr);

  for (int i = 0; i < 5; ++i) {
    allocations.emplace_back(std::make_shared<rmm::device_buffer>(10_MiB));
  }

  // Check the allocated bytes for both MRs
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 15);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 5);

  EXPECT_EQ(mr.get_allocated_bytes(), 150_MiB);
  EXPECT_EQ(inner_mr.get_allocated_bytes(), 50_MiB);

  // Clear the allocations, causing all memory to be freed
  allocations.clear();

  // The current allocations for both MRs should be 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);

  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(inner_mr.get_allocated_bytes(), 0);

  // Finally, verify the peak and total values
  EXPECT_EQ(mr.get_total_allocation_counts().peak_bytes, 150_MiB);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().peak_bytes, 50_MiB);

  EXPECT_EQ(mr.get_total_allocation_counts().peak_count, 15);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().peak_count, 5);

  // Reset the current device resource
  rmm::mr::set_current_device_resource(mr.get_upstream());
}

TEST(TrackingTest, NegativeInnerTracking)
{
  // This tests the unlikely scenario where pointers are deallocated on an inner
  // wrapped memory resource. This can happen if the MR is not saved with the
  // memory pointer
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 10);

  tracking_adaptor inner_mr{&mr};

  // Add more allocations
  for (int i = 0; i < 5; ++i) {
    allocations.push_back(inner_mr.allocate(10_MiB));
  }

  // Check the outstanding allocations
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 15);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 5);

  // Check the current counts
  EXPECT_EQ(mr.get_total_allocation_counts().current_bytes, 150_MiB);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().current_bytes, 50_MiB);

  EXPECT_EQ(mr.get_total_allocation_counts().current_count, 15);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().current_count, 5);

  // Deallocate all allocations using the inner_mr
  for (int i = 0; i < allocations.size(); ++i) {
    inner_mr.deallocate(allocations[i], 10_MiB);
  }
  allocations.clear();

  // Check the outstanding allocations are all 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);

  // Check the current counts are 0 for the outer
  EXPECT_EQ(mr.get_total_allocation_counts().current_bytes, 0);
  EXPECT_EQ(mr.get_total_allocation_counts().current_count, 0);

  // The inner_mr will have negative values
  EXPECT_EQ(inner_mr.get_total_allocation_counts().current_bytes, -100_MiB);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().current_count, -10);

  // Verify the peak and total
  EXPECT_EQ(mr.get_total_allocation_counts().peak_bytes, 150_MiB);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().peak_bytes, 50_MiB);

  EXPECT_EQ(mr.get_total_allocation_counts().peak_count, 15);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().peak_count, 5);

  EXPECT_EQ(mr.get_total_allocation_counts().total_bytes, 150_MiB);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().total_bytes, 50_MiB);

  EXPECT_EQ(mr.get_total_allocation_counts().total_count, 15);
  EXPECT_EQ(inner_mr.get_total_allocation_counts().total_count, 5);
}

TEST(TrackingTest, DeallocWrongBytes)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  // When deallocating, pass the wrong bytes to deallocate
  for (int i = 0; i < allocations.size(); ++i) {
    mr.deallocate(allocations[i], 5_MiB);
  }
  allocations.clear();

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);

  // allocation_counts should be unaffected
  auto current_alloc_counts = mr.get_total_allocation_counts();

  // Verify current allocations are correct despite the error
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(current_alloc_counts.current_bytes, 0);
  EXPECT_EQ(current_alloc_counts.current_count, 0);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_counts.peak_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 10);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 10);
}

TEST(TrackingTest, PushPop)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void *> allocations;

  // Up to 90
  for (int i = 0; i < 9; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  // Go down to 20 remaining
  for (auto &&it = allocations.begin(); it != allocations.end();) {
    mr.deallocate(*it, 10_MiB);
    it = allocations.erase(it);

    if (allocations.size() == 2) { break; }
  }

  // Push and verify counts
  auto current_alloc_counts = mr.push_allocation_counts();

  // Verify current allocations
  EXPECT_EQ(mr.get_allocated_bytes(), 20_MiB);
  EXPECT_EQ(current_alloc_counts.current_bytes, 20_MiB);
  EXPECT_EQ(current_alloc_counts.current_count, 2);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_counts.peak_bytes, 90_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 9);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 90_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 9);

  // Now allocate up to 100. This should result in index 0 with a peak of 90,
  // index 1 with a peak of 80 and the total with a peak of 100
  while (allocations.size() < 10) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  // Before popping Verify the total with 10 allocated
  current_alloc_counts = mr.get_total_allocation_counts();

  // Verify current allocations
  EXPECT_EQ(mr.get_allocated_bytes(), 100_MiB);
  EXPECT_EQ(current_alloc_counts.current_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.current_count, 10);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_counts.peak_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 10);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 170_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 17);

  // Deallocate all remaining
  for (int i = 0; i < allocations.size(); ++i) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  allocations.clear();

  // Pop the most recent
  current_alloc_counts = mr.pop_allocation_counts();

  // Verify the popped
  EXPECT_EQ(current_alloc_counts.current_bytes, -20_MiB);
  EXPECT_EQ(current_alloc_counts.current_count, -2);

  EXPECT_EQ(current_alloc_counts.peak_bytes, 80_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 8);

  EXPECT_EQ(current_alloc_counts.total_bytes, 80_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 8);

  // Check the total one more time to ensure its the same
  current_alloc_counts = mr.get_total_allocation_counts();

  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(current_alloc_counts.current_bytes, 0);
  EXPECT_EQ(current_alloc_counts.current_count, 0);

  EXPECT_EQ(current_alloc_counts.peak_bytes, 100_MiB);
  EXPECT_EQ(current_alloc_counts.peak_count, 10);

  // Verify total allocations
  EXPECT_EQ(current_alloc_counts.total_bytes, 170_MiB);
  EXPECT_EQ(current_alloc_counts.total_count, 17);
}

TEST(TrackingTest, OOBPopAllocCounts)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};

  mr.push_allocation_counts();
  mr.pop_allocation_counts();

  EXPECT_THROW(mr.pop_allocation_counts(), rmm::out_of_range);
}

}  // namespace
}  // namespace test
}  // namespace rmm
