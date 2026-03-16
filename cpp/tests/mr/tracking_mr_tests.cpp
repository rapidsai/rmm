/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {

using tracking_adaptor = rmm::mr::tracking_resource_adaptor<rmm::mr::device_memory_resource>;

constexpr auto num_allocations{10};
constexpr auto num_more_allocations{5};
constexpr auto ten_MiB{10_MiB};

class delayed_memory_resource {
 public:
  delayed_memory_resource(rmm::device_async_resource_ref upstream, std::chrono::milliseconds delay)
    : upstream_{upstream}, delay_{delay}
  {
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment)
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    std::this_thread::sleep_for(delay_);
  }
  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate(stream, bytes, alignment);
  }
  void deallocate(rmm::cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment)
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    std::this_thread::sleep_for(delay_);
  }
  friend void get_property(delayed_memory_resource const&, cuda::mr::device_accessible) noexcept {}
  bool operator==(delayed_memory_resource const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(delayed_memory_resource const& other) const noexcept
  {
    return !(this == std::addressof(other));
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::chrono::milliseconds delay_;
};
static_assert(cuda::mr::resource<delayed_memory_resource>);
static_assert(cuda::mr::resource_with<delayed_memory_resource, cuda::mr::device_accessible>);

TEST(TrackingTest, MultiThreaded)
{
  auto upstream = rmm::mr::cuda_memory_resource{};
  std::vector<std::thread> threads;
  auto delayed = delayed_memory_resource(upstream, std::chrono::milliseconds{300});
  auto mr      = rmm::mr::tracking_resource_adaptor<delayed_memory_resource>(delayed);
  auto stream  = rmm::cuda_stream{};
  // Idea, we want to provoke address reuse to test ABA problems in the tracking resource
  // adaptor. To do so, the delayed memory resource frees (and hence returns to the
  // upstream) an address immediately and then makes that thread sleep. So thread 0
  // allocates, deallocates, sleeps. Thread 1 sleeps, allocates, deallocates, sleeps. We
  // therefore expect an interleaving:
  //
  // Thread-0             Thread-1
  // alloc
  // dealloc-start
  //                      alloc
  //                      dealloc-start
  //
  // dealloc-end
  //                      dealloc-end
  //
  // In this scenario, if the tracking adaptor doesn't correctly handle ordering,
  // allocation tracking should be morally an acquire-release pair bounded by the upstream
  // allocate/deallocate, then we can get ABA reuse of the upstream's pointer.
  for (int i = 0; i < 2; i++) {
    threads.emplace_back([&, i = i]() {
      if (i == 0) {
        void* ptr{nullptr};
        EXPECT_NO_THROW(ptr = mr.allocate(stream, 256, rmm::CUDA_ALLOCATION_ALIGNMENT));
        EXPECT_NE(ptr, nullptr);
        mr.deallocate(stream, ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        void* ptr{nullptr};
        EXPECT_NO_THROW(ptr = mr.allocate(stream, 256, rmm::CUDA_ALLOCATION_ALIGNMENT));
        EXPECT_NE(ptr, nullptr);
        mr.deallocate(stream, ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST(TrackingTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { tracking_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(TrackingTest, Empty)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, AllFreed)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  allocations.reserve(num_allocations);
  for (int i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }
  for (auto* alloc : allocations) {
    mr.deallocate_sync(alloc, ten_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, AllocationsLeftWithStacks)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref(), true};
  std::vector<void*> allocations;
  allocations.reserve(num_allocations);
  for (int i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }
  for (int i = 0; i < num_allocations; i += 2) {
    mr.deallocate_sync(allocations[i], ten_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations / 2);
  EXPECT_EQ(mr.get_allocated_bytes(), ten_MiB * (num_allocations / 2));
  auto const& outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), num_allocations / 2);
  EXPECT_NE(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, AllocationsLeftWithoutStacks)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  allocations.reserve(num_allocations);
  for (int i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  for (int i = 0; i < num_allocations; i += 2) {
    mr.deallocate_sync(allocations[i], ten_MiB);
  }
  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations / 2);
  EXPECT_EQ(mr.get_allocated_bytes(), ten_MiB * (num_allocations / 2));
  auto const& outstanding_allocations = mr.get_outstanding_allocations();
  EXPECT_EQ(outstanding_allocations.size(), num_allocations / 2);
  EXPECT_EQ(outstanding_allocations.begin()->second.strace, nullptr);
}

TEST(TrackingTest, MultiTracking)
{
  // Test stacking multiple tracking adaptors, using explicit resource refs
  // to avoid lifetime issues with the global device resource map
  auto orig_device_resource = rmm::mr::get_current_device_resource_ref();
  tracking_adaptor mr{orig_device_resource, true};

  std::vector<std::shared_ptr<rmm::device_buffer>> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(ten_MiB, rmm::cuda_stream_default, &mr));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations);

  tracking_adaptor inner_mr{&mr};

  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.emplace_back(
      std::make_shared<rmm::device_buffer>(ten_MiB, rmm::cuda_stream_default, &inner_mr));
  }

  // Check the allocated bytes for both MRs
  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations + num_more_allocations);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), num_more_allocations);

  EXPECT_EQ(mr.get_allocated_bytes(), ten_MiB * (num_allocations + num_more_allocations));
  EXPECT_EQ(inner_mr.get_allocated_bytes(), ten_MiB * num_more_allocations);

  EXPECT_GT(mr.get_outstanding_allocations_str().size(), 0);

  // Clear the allocations, causing all memory to be freed
  allocations.clear();

  // The current allocations for both MRs should be 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);

  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(inner_mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, NegativeInnerTracking)
{
  // This tests the unlikely scenario where pointers are deallocated on an inner
  // wrapped memory resource. This can happen if the MR is not saved with the
  // memory pointer
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations);

  tracking_adaptor inner_mr{&mr};

  // Add more allocations
  for (std::size_t i = 0; i < num_more_allocations; ++i) {
    allocations.push_back(inner_mr.allocate_sync(ten_MiB));
  }

  // Check the outstanding allocations
  EXPECT_EQ(mr.get_outstanding_allocations().size(), num_allocations + num_more_allocations);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), num_more_allocations);

  // Deallocate all allocations using the inner_mr
  for (auto& allocation : allocations) {
    inner_mr.deallocate_sync(allocation, ten_MiB);
  }
  allocations.clear();

  // Check the outstanding allocations are all 0
  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(inner_mr.get_outstanding_allocations().size(), 0);
}

TEST(TrackingTest, DeallocWrongBytes)
{
  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  // When deallocating, pass the wrong bytes to deallocate
  for (auto& allocation : allocations) {
    mr.deallocate_sync(allocation, ten_MiB / 2);
  }
  allocations.clear();

  EXPECT_EQ(mr.get_outstanding_allocations().size(), 0);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);

  // Verify current allocations are correct despite the error
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(TrackingTest, LogOutstandingAllocations)
{
  std::ostringstream oss;
  auto oss_sink  = std::make_shared<rapids_logger::ostream_sink_mt>(oss);
  auto old_level = rmm::default_logger().level();
  rmm::default_logger().sinks().push_back(oss_sink);

  tracking_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  std::vector<void*> allocations;
  for (std::size_t i = 0; i < num_allocations; ++i) {
    allocations.push_back(mr.allocate_sync(ten_MiB));
  }

  rmm::default_logger().set_level(rapids_logger::level_enum::debug);
  EXPECT_NO_THROW(mr.log_outstanding_allocations());

#if RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
  EXPECT_NE(oss.str().find("Outstanding Allocations"), std::string::npos);
#endif

  for (auto& allocation : allocations) {
    mr.deallocate_sync(allocation, ten_MiB);
  }

  rmm::default_logger().set_level(old_level);
  rmm::default_logger().sinks().pop_back();
}

}  // namespace
}  // namespace rmm::test
