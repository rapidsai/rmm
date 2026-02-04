/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mr_ref_test.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace rmm::test {
namespace {

// Helper functions for multi-threaded tests
template <typename Task, typename... Arguments>
void spawn_n(std::size_t num_threads, Task task, Arguments&&... args)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread(task, std::forward<Arguments>(args)...));
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

template <typename Task, typename... Arguments>
void spawn(Task task, Arguments&&... args)
{
  spawn_n(4, task, std::forward<Arguments>(args)...);
}

// Single-threaded default resource tests

TEST(DefaultTest, CurrentDeviceResourceIsCUDA)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
  EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
#pragma GCC diagnostic pop
}

TEST(DefaultTest, UseCurrentDeviceResource) { test_get_current_device_resource(); }

TEST(DefaultTest, UseCurrentDeviceResourceRef) { test_get_current_device_resource_ref(); }

TEST(DefaultTest, GetCurrentDeviceResource)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto* mr = rmm::mr::get_current_device_resource();
#pragma GCC diagnostic pop
  EXPECT_NE(nullptr, mr);
  EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
}

TEST(DefaultTest, GetCurrentDeviceResourceRef)
{
  auto mr = rmm::mr::get_current_device_resource_ref();
  EXPECT_EQ(mr, rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
}

TEST(DefaultTest, SetCurrentDeviceResourceRefFromPointer)
{
  // Construct a cuda_memory_resource
  rmm::mr::cuda_memory_resource cuda_mr{};

  // Get a pointer to it (device_memory_resource*)
  rmm::mr::device_memory_resource* mr_ptr = &cuda_mr;

  // Set with set_current_device_resource_ref using the pointer
  rmm::mr::set_current_device_resource_ref(mr_ptr);

  // Get the ref with get_current_device_resource_ref
  auto ref = rmm::mr::get_current_device_resource_ref();

  // Use that ref to allocate
  constexpr std::size_t size{1024};
  void* ptr = ref.allocate_sync(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(is_properly_aligned(ptr));
  EXPECT_TRUE(is_device_accessible_memory(ptr));

  // Deallocate
  ref.deallocate_sync(ptr, size);

  // Reset to initial resource
  rmm::mr::reset_current_device_resource_ref();
}

// Multi-threaded default resource tests

TEST(DefaultTest, UseCurrentDeviceResource_mt) { spawn(test_get_current_device_resource); }

TEST(DefaultTest, UseCurrentDeviceResourceRef_mt) { spawn(test_get_current_device_resource_ref); }

TEST(DefaultTest, CurrentDeviceResourceIsCUDA_mt)
{
  spawn([]() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
    EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
#pragma GCC diagnostic pop
  });
}

TEST(DefaultTest, CurrentDeviceResourceRefIsCUDA_mt)
{
  spawn([]() {
    EXPECT_EQ(rmm::mr::get_current_device_resource_ref(),
              rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
  });
}

TEST(DefaultTest, GetCurrentDeviceResource_mt)
{
  spawn([]() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
#pragma GCC diagnostic pop
    EXPECT_NE(nullptr, mr);
    EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
  });
}

TEST(DefaultTest, GetCurrentDeviceResourceRef_mt)
{
  spawn([]() {
    auto mr = rmm::mr::get_current_device_resource_ref();
    EXPECT_EQ(mr, rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
  });
}

}  // namespace
}  // namespace rmm::test
