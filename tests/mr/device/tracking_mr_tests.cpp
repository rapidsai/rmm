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

#include <rmm/cuda_stream_view.hpp>
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
  std::vector<void*> allocations;
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
  std::vector<void*> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  for (int i = 0; i < 10; i += 2) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 5);
  EXPECT_EQ(mr.get_allocated_bytes(), 50_MiB);
  auto const& outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), 5);
  EXPECT_NE(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, AllocationsLeftWithoutStacks)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void*> allocations;
  for (int i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }
  for (int i = 0; i < 10; i += 2) {
    mr.deallocate(allocations[i], 10_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 5);
  EXPECT_EQ(mr.get_allocated_bytes(), 50_MiB);
  auto const& outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), 5);
  EXPECT_EQ(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, MultiTracking)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource(), true};
  rmm::mr::set_current_device_resource(&mr);

  std::vector<std::shared_ptr<rmm::device_buffer>> allocations;
  for (std::size_t i = 0; i < 10; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(10_MiB, rmm::cuda_stream_default));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 10);

  tracking_adaptor inner_mr{rmm::mr::get_current_device_resource()};
  rmm::mr::set_current_device_resource(&inner_mr);

  for (std::size_t i = 0; i < 5; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(10_MiB, rmm::cuda_stream_default));
  }

  // Check the allocated bytes for both MRs
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 15);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 5);

  EXPECT_EQ(mr.get_allocated_bytes(), 150_MiB);
  EXPECT_EQ(inner_mr.get_allocated_bytes(), 50_MiB);

  EXPECT_GT(mr.get_outstanding_allocations_str().size(), 0);

  // Clear the allocations, causing all memory to be freed
  allocations.clear();

  // The current allocations for both MRs should be 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);

  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(inner_mr.get_allocated_bytes(), 0);

  // Reset the current device resource
  rmm::mr::set_current_device_resource(mr.get_upstream());
}

TEST(TrackingTest, NegativeInnerTracking)
{
  // This tests the unlikely scenario where pointers are deallocated on an inner
  // wrapped memory resource. This can happen if the MR is not saved with the
  // memory pointer
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 10);

  tracking_adaptor inner_mr{&mr};

  // Add more allocations
  for (std::size_t i = 0; i < 5; ++i) {
    allocations.push_back(inner_mr.allocate(10_MiB));
  }

  // Check the outstanding allocations
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 15);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 5);

  // Deallocate all allocations using the inner_mr
  for (std::size_t i = 0; i < allocations.size(); ++i) {
    inner_mr.deallocate(allocations[i], 10_MiB);
  }
  allocations.clear();

  // Check the outstanding allocations are all 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);
}

TEST(TrackingTest, DeallocWrongBytes)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < 10; ++i) {
    allocations.push_back(mr.allocate(10_MiB));
  }

  // When deallocating, pass the wrong bytes to deallocate
  for (std::size_t i = 0; i < allocations.size(); ++i) {
    mr.deallocate(allocations[i], 5_MiB);
  }
  allocations.clear();

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);

  // Verify current allocations are correct despite the error
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

}  // namespace
}  // namespace test
}  // namespace rmm
