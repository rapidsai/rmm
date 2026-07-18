/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"
#include "../mock_resource.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>

namespace rmm::test {
namespace {

using ::testing::_;

TEST(CallbackTest, TestCallbacksAreInvoked)
{
  auto const size      = std::size_t{10_MiB};
  auto const alignment = std::size_t{128};
  auto base_mr         = mock_resource();
  auto base_wrapper    = mock_resource_wrapper{&base_mr};
  auto base_ref        = device_async_resource_ref{base_wrapper};
  EXPECT_CALL(base_mr, allocate(_, size, alignment)).Times(1);
  EXPECT_CALL(base_mr, deallocate(_, _, size, alignment)).Times(1);

  auto allocate_callback =
    [](cuda_stream_view stream, std::size_t size, std::size_t alignment, void* arg) {
      auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
      return base_mr.allocate(stream, size, alignment);
    };
  auto deallocate_callback =
    [](cuda_stream_view stream, void* ptr, std::size_t size, std::size_t alignment, void* arg) {
      auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
      base_mr.deallocate(stream, ptr, size, alignment);
    };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, &base_ref, &base_ref);
  auto* ptr = mr.allocate_sync(size, alignment);
  mr.deallocate_sync(ptr, size, alignment);
}

TEST(CallbackTest, ForwardsAllocationMetadata)
{
  auto base_mr         = rmm::mr::get_current_device_resource_ref();
  auto stream          = rmm::cuda_stream{};
  auto const size      = std::size_t{1024};
  auto const alignment = std::size_t{128};

  cuda_stream_view allocation_stream{};
  cuda_stream_view deallocation_stream{};
  void* allocation_ptr{};
  void* deallocation_ptr{};
  std::size_t allocation_size{};
  std::size_t deallocation_size{};
  std::size_t allocation_alignment{};
  std::size_t deallocation_alignment{};

  auto allocate_callback = [&](cuda_stream_view callback_stream,
                               std::size_t callback_size,
                               std::size_t callback_alignment,
                               void*) {
    allocation_stream    = callback_stream;
    allocation_size      = callback_size;
    allocation_alignment = callback_alignment;
    allocation_ptr       = base_mr.allocate(callback_stream, callback_size, callback_alignment);
    return allocation_ptr;
  };
  auto deallocate_callback = [&](cuda_stream_view callback_stream,
                                 void* ptr,
                                 std::size_t callback_size,
                                 std::size_t callback_alignment,
                                 void*) {
    deallocation_stream    = callback_stream;
    deallocation_ptr       = ptr;
    deallocation_size      = callback_size;
    deallocation_alignment = callback_alignment;
    base_mr.deallocate(callback_stream, ptr, callback_size, callback_alignment);
  };
  auto mr = rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback);

  auto* ptr = mr.allocate(stream, size, alignment);
  mr.deallocate(stream, ptr, size, alignment);

  EXPECT_EQ(allocation_stream, stream);
  EXPECT_EQ(allocation_ptr, ptr);
  EXPECT_EQ(allocation_size, size);
  EXPECT_EQ(allocation_alignment, alignment);
  EXPECT_EQ(deallocation_stream, allocation_stream);
  EXPECT_EQ(deallocation_ptr, allocation_ptr);
  EXPECT_EQ(deallocation_size, allocation_size);
  EXPECT_EQ(deallocation_alignment, allocation_alignment);
}

TEST(CallbackTest, RejectsInvalidAllocationAlignmentBeforeCallback)
{
  std::size_t callback_invocations{};
  auto allocate_callback = [&callback_invocations](
                             cuda_stream_view, std::size_t, std::size_t, void*) {
    ++callback_invocations;
    return nullptr;
  };
  auto deallocate_callback = [](cuda_stream_view, void*, std::size_t, std::size_t, void*) {};
  auto mr = rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback);

  EXPECT_THROW((void)mr.allocate_sync(1024, 3), rmm::logic_error);
  EXPECT_EQ(callback_invocations, 0);
}

TEST(CallbackTest, LoggingTest)
{
  testing::internal::CaptureStdout();

  auto base_mr = rmm::mr::get_current_device_resource_ref();
  auto allocate_callback =
    [](cuda_stream_view stream, std::size_t size, std::size_t alignment, void* arg) {
      std::cout << "Allocating " << size << " bytes" << std::endl;
      auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
      return base_mr.allocate(stream, size, alignment);
    };

  auto deallocate_callback =
    [](cuda_stream_view stream, void* ptr, std::size_t size, std::size_t alignment, void* arg) {
      std::cout << "Deallocating " << size << " bytes" << std::endl;
      auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
      base_mr.deallocate(stream, ptr, size, alignment);
    };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, &base_mr, &base_mr);
  auto const size = std::size_t{10_MiB};
  auto* ptr       = mr.allocate_sync(size);
  mr.deallocate_sync(ptr, size);

  auto output = testing::internal::GetCapturedStdout();
  auto expect = std::string("Allocating ") + std::to_string(size) + " bytes\nDeallocating " +
                std::to_string(size) + " bytes\n";
  ASSERT_EQ(expect, output);
}

}  // namespace
}  // namespace rmm::test
