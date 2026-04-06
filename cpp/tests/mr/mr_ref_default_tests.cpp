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

TEST(DefaultTest, UseCurrentDeviceResourceRef) { test_get_current_device_resource_ref(); }

TEST(DefaultTest, GetCurrentDeviceResourceRef)
{
  auto mr = rmm::mr::get_current_device_resource_ref();
  EXPECT_EQ(mr, rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
}

TEST(DefaultTest, SetCurrentDeviceResourceRef)
{
  rmm::mr::cuda_memory_resource cuda_mr{};

  rmm::mr::set_current_device_resource_ref(cuda_mr);

  auto ref = rmm::mr::get_current_device_resource_ref();

  constexpr std::size_t size{1024};
  void* ptr = ref.allocate_sync(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(is_properly_aligned(ptr));
  EXPECT_TRUE(is_device_accessible_memory(ptr));

  ref.deallocate_sync(ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  rmm::mr::reset_current_device_resource_ref();
}

// Multi-threaded default resource tests

TEST(DefaultTest, UseCurrentDeviceResourceRef_mt) { spawn(test_get_current_device_resource_ref); }

TEST(DefaultTest, CurrentDeviceResourceRefIsCUDA_mt)
{
  spawn([]() {
    EXPECT_EQ(rmm::mr::get_current_device_resource_ref(),
              rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()});
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
