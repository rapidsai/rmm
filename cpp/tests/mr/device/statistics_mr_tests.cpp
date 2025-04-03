/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rmm::test {
namespace {

using statistics_adaptor = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

constexpr auto num_allocations{10};
constexpr auto num_more_allocations{5};
constexpr auto ten_MiB{10_MiB};

TEST(StatisticsTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { statistics_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(StatisticsTest, Empty)
{
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};

  EXPECT_EQ(mr.get_bytes_counter().peak, 0);
  EXPECT_EQ(mr.get_bytes_counter().total, 0);
  EXPECT_EQ(mr.get_bytes_counter().value, 0);

  EXPECT_EQ(mr.get_allocations_counter().peak, 0);
  EXPECT_EQ(mr.get_allocations_counter().total, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
}

TEST(StatisticsTest, AllFreed)
{
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;

  allocations.reserve(num_allocations);
  for (int i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate(ten_MiB));
  }
  for (auto* alloc : allocations) {
    mr.deallocate(alloc, ten_MiB);
  }

  // Counter values should be 0
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
}

TEST(StatisticsTest, PeakAllocations)
{
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;

  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate(ten_MiB));
  }
  // Delete every other allocation
  for (auto&& it = allocations.begin(); it != allocations.end(); ++it) {
    mr.deallocate(*it, ten_MiB);
    it = allocations.erase(it);
  }

  auto current_alloc_counts = mr.get_allocations_counter();
  auto current_alloc_bytes  = mr.get_bytes_counter();

  // Verify current allocations
  EXPECT_EQ(current_alloc_bytes.value, 50_MiB);
  EXPECT_EQ(current_alloc_counts.value, 5);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_bytes.peak, 100_MiB);
  EXPECT_EQ(current_alloc_counts.peak, 10);

  // Verify total allocations
  EXPECT_EQ(current_alloc_bytes.total, 100_MiB);
  EXPECT_EQ(current_alloc_counts.total, 10);

  // Add 10 more to increase the peak
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate(ten_MiB));
  }

  // Deallocate all remaining
  for (auto& allocation : allocations) {
    mr.deallocate(allocation, ten_MiB);
  }
  allocations.clear();

  current_alloc_counts = mr.get_allocations_counter();
  current_alloc_bytes  = mr.get_bytes_counter();

  // Verify current allocations
  EXPECT_EQ(current_alloc_bytes.value, 0);
  EXPECT_EQ(current_alloc_counts.value, 0);

  // Verify peak allocations
  EXPECT_EQ(current_alloc_bytes.peak, 150_MiB);
  EXPECT_EQ(current_alloc_counts.peak, 15);

  // Verify total allocations
  EXPECT_EQ(current_alloc_bytes.total, 200_MiB);
  EXPECT_EQ(current_alloc_counts.total, 20);
}

TEST(StatisticsTest, MultiTracking)
{
  auto orig_device_resource = rmm::mr::get_current_device_resource_ref();
  statistics_adaptor mr{orig_device_resource};
  rmm::mr::set_current_device_resource_ref(mr);

  std::vector<std::shared_ptr<rmm::device_buffer>> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(ten_MiB, rmm::cuda_stream_default));
  }

  EXPECT_EQ(mr.get_allocations_counter().value, 10);

  statistics_adaptor inner_mr{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::set_current_device_resource_ref(inner_mr);

  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(ten_MiB, rmm::cuda_stream_default));
  }

  // Check the allocated bytes for both MRs
  EXPECT_EQ(mr.get_allocations_counter().value, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().value, 5);

  EXPECT_EQ(mr.get_bytes_counter().value, 150_MiB);
  EXPECT_EQ(inner_mr.get_bytes_counter().value, 50_MiB);

  // Clear the allocations, causing all memory to be freed
  allocations.clear();

  // The current allocations for both MRs should be 0
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  EXPECT_EQ(inner_mr.get_allocations_counter().value, 0);

  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(inner_mr.get_bytes_counter().value, 0);

  // Finally, verify the peak and total values
  EXPECT_EQ(mr.get_bytes_counter().peak, 150_MiB);
  EXPECT_EQ(inner_mr.get_bytes_counter().peak, 50_MiB);

  EXPECT_EQ(mr.get_allocations_counter().peak, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().peak, 5);

  // Reset the current device resource
  rmm::mr::set_current_device_resource_ref(orig_device_resource);
}

TEST(StatisticsTest, NegativeInnerTracking)
{
  // This tests the unlikely scenario where pointers are deallocated on an inner
  // wrapped memory resource. This can happen if the MR is not saved with the
  // memory pointer
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate(ten_MiB));
  }

  EXPECT_EQ(mr.get_allocations_counter().value, 10);

  statistics_adaptor inner_mr{&mr};

  // Add more allocations
  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.push_back(inner_mr.allocate(ten_MiB));
  }

  // Check the outstanding allocations
  EXPECT_EQ(mr.get_allocations_counter().value, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().value, 5);

  // Check the current counts
  EXPECT_EQ(mr.get_bytes_counter().value, 150_MiB);
  EXPECT_EQ(inner_mr.get_bytes_counter().value, 50_MiB);

  EXPECT_EQ(mr.get_allocations_counter().value, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().value, 5);

  // Deallocate all allocations using the inner_mr
  for (auto& allocation : allocations) {
    inner_mr.deallocate(allocation, ten_MiB);
  }
  allocations.clear();

  // Check the current counts are 0 for the outer
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);

  // The inner_mr will have negative values
  EXPECT_EQ(inner_mr.get_bytes_counter().value, -100_MiB);
  EXPECT_EQ(inner_mr.get_allocations_counter().value, -10);

  // Verify the peak and total
  EXPECT_EQ(mr.get_bytes_counter().peak, 150_MiB);
  EXPECT_EQ(inner_mr.get_bytes_counter().peak, 50_MiB);

  EXPECT_EQ(mr.get_allocations_counter().peak, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().peak, 5);

  EXPECT_EQ(mr.get_bytes_counter().total, 150_MiB);
  EXPECT_EQ(inner_mr.get_bytes_counter().total, 50_MiB);

  EXPECT_EQ(mr.get_allocations_counter().total, 15);
  EXPECT_EQ(inner_mr.get_allocations_counter().total, 5);
}

TEST(StatisticsTest, Nested)
{
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  void* a0 = mr.allocate(ten_MiB);
  EXPECT_EQ(mr.get_bytes_counter().value, ten_MiB);
  EXPECT_EQ(mr.get_allocations_counter().value, 1);
  {
    auto [bytes, allocs] = mr.push_counters();
    EXPECT_EQ(bytes.value, ten_MiB);
    EXPECT_EQ(allocs.value, 1);
  }
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  void* a1 = mr.allocate(ten_MiB);
  mr.push_counters();
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  void* a2 = mr.allocate(ten_MiB);
  mr.deallocate(a2, ten_MiB);
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_bytes_counter().peak, ten_MiB);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().peak, 1);
  {
    auto [bytes, allocs] = mr.pop_counters();
    EXPECT_EQ(bytes.value, 0);
    EXPECT_EQ(bytes.peak, ten_MiB);
    EXPECT_EQ(allocs.value, 0);
    EXPECT_EQ(allocs.peak, 1);
  }
  mr.deallocate(a0, ten_MiB);
  {
    auto [bytes, allocs] = mr.pop_counters();
    EXPECT_EQ(bytes.value, 0);
    EXPECT_EQ(bytes.peak, ten_MiB * 2);
    EXPECT_EQ(allocs.value, 0);
    EXPECT_EQ(allocs.peak, 2);
  }
  mr.deallocate(a1, ten_MiB);
  EXPECT_THROW(mr.pop_counters(), std::out_of_range);
}

}  // namespace
}  // namespace rmm::test
