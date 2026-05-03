/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"
#include "delayed_memory_resource.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {

using statistics_adaptor = rmm::mr::statistics_resource_adaptor;

constexpr auto num_allocations{10};
constexpr auto num_more_allocations{5};
constexpr auto ten_MiB{10_MiB};

struct allocation_size : public ::testing::TestWithParam<std::size_t> {};

INSTANTIATE_TEST_SUITE_P(StatisticsTest, allocation_size, ::testing::Values(0, 256));

TEST_P(allocation_size, MultiThreaded)
{
  const std::size_t allocation_size = GetParam();
  auto upstream                     = rmm::mr::cuda_memory_resource{};
  auto delayed = delayed_memory_resource(upstream, std::chrono::milliseconds{300});
  statistics_adaptor mr{rmm::device_async_resource_ref{delayed}};
  auto stream = rmm::cuda_stream{};
  // Provoke interleaving to test that statistics counters are updated with correct ordering
  // relative to upstream deallocate. The delayed memory resource frees the pointer upstream
  // immediately then sleeps, simulating the window where the address is available for reuse
  // but the adaptor hasn't updated its counters yet.
  //
  // Thread-0             Thread-1
  // alloc
  //                      alloc
  //                      dealloc-start
  // dealloc-start
  //                      dealloc-end
  // dealloc-end
  //
  // After both threads complete, the counters must reflect zero outstanding allocations.
  std::vector<std::thread> threads;
  for (int i = 0; i < 2; i++) {
    threads.emplace_back([&, i = i]() {
      void* ptr{nullptr};
      if (i != 0) { std::this_thread::sleep_for(std::chrono::milliseconds{100}); }
      EXPECT_NO_THROW(ptr = mr.allocate(stream, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT));
      if (allocation_size != 0) {
        EXPECT_NE(ptr, nullptr);
      } else {
        EXPECT_EQ(ptr, nullptr);
      }
      if (i == 0) { std::this_thread::sleep_for(std::chrono::milliseconds{100}); }
      mr.deallocate(stream, ptr, allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().total, 2);
  EXPECT_EQ(mr.get_bytes_counter().total, 2 * allocation_size);
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
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }
  for (auto* alloc : allocations) {
    mr.deallocate_sync(alloc, ten_MiB);
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
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }
  // Delete every other allocation
  for (auto&& it = allocations.begin(); it != allocations.end(); ++it) {
    mr.deallocate_sync(*it, ten_MiB);
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
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  // Deallocate all remaining
  for (auto& allocation : allocations) {
    mr.deallocate_sync(allocation, ten_MiB);
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
  // Test stacking multiple statistics adaptors, using explicit resource refs
  // to avoid lifetime issues with the global device resource map
  auto orig_device_resource = rmm::mr::get_current_device_resource_ref();
  statistics_adaptor mr{orig_device_resource};

  std::vector<std::shared_ptr<rmm::device_buffer>> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(ten_MiB, cuda::stream_ref{cudaStream_t{nullptr}}, mr));
  }

  EXPECT_EQ(mr.get_allocations_counter().value, 10);

  statistics_adaptor inner_mr{rmm::device_async_resource_ref{mr}};

  rmm::device_async_resource_ref inner_ref{inner_mr};
  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.emplace_back(std::make_shared<rmm::device_buffer>(
      ten_MiB, cuda::stream_ref{cudaStream_t{nullptr}}, inner_ref));
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
}

TEST(StatisticsTest, NegativeInnerTracking)
{
  // This tests the unlikely scenario where pointers are deallocated on an inner
  // wrapped memory resource. This can happen if the MR is not saved with the
  // memory pointer
  statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  EXPECT_EQ(mr.get_allocations_counter().value, 10);

  statistics_adaptor inner_mr{rmm::device_async_resource_ref{mr}};

  // Add more allocations
  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.push_back(inner_mr.allocate_sync(ten_MiB));
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
    inner_mr.deallocate_sync(allocation, ten_MiB);
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
  void* a0 = mr.allocate_sync(ten_MiB);
  EXPECT_EQ(mr.get_bytes_counter().value, ten_MiB);
  EXPECT_EQ(mr.get_allocations_counter().value, 1);
  {
    auto [bytes, allocs] = mr.push_counters();
    EXPECT_EQ(bytes.value, ten_MiB);
    EXPECT_EQ(allocs.value, 1);
  }
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  void* a1 = mr.allocate_sync(ten_MiB);
  mr.push_counters();
  EXPECT_EQ(mr.get_bytes_counter().value, 0);
  EXPECT_EQ(mr.get_allocations_counter().value, 0);
  void* a2 = mr.allocate_sync(ten_MiB);
  mr.deallocate_sync(a2, ten_MiB);
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
  mr.deallocate_sync(a0, ten_MiB);
  {
    auto [bytes, allocs] = mr.pop_counters();
    EXPECT_EQ(bytes.value, 0);
    EXPECT_EQ(bytes.peak, ten_MiB * 2);
    EXPECT_EQ(allocs.value, 0);
    EXPECT_EQ(allocs.peak, 2);
  }
  mr.deallocate_sync(a1, ten_MiB);
  EXPECT_THROW(mr.pop_counters(), std::out_of_range);
}

}  // namespace
}  // namespace rmm::test
