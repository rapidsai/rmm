/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "../../mock_resource.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/callback_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
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
  auto base_mr  = mock_resource();
  auto base_ref = device_async_resource_ref{base_mr};
  EXPECT_CALL(base_mr, do_allocate(10_MiB, cuda_stream_view{})).Times(1);
  EXPECT_CALL(base_mr, do_deallocate(_, 10_MiB, cuda_stream_view{})).Times(1);

  auto allocate_callback = [](std::size_t size, cuda_stream_view stream, void* arg) {
    auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
    return base_mr.allocate_async(size, stream);
  };
  auto deallocate_callback = [](void* ptr, std::size_t size, cuda_stream_view stream, void* arg) {
    auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
    base_mr.deallocate_async(ptr, size, stream);
  };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, &base_ref, &base_ref);
  auto const size = std::size_t{10_MiB};
  auto* ptr       = mr.allocate(size);
  mr.deallocate(ptr, size);
}

TEST(CallbackTest, LoggingTest)
{
  testing::internal::CaptureStdout();

  auto base_mr           = rmm::mr::get_current_device_resource_ref();
  auto allocate_callback = [](std::size_t size, cuda_stream_view stream, void* arg) {
    std::cout << "Allocating " << size << " bytes" << std::endl;
    auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
    return base_mr.allocate_async(size, stream);
  };

  auto deallocate_callback = [](void* ptr, std::size_t size, cuda_stream_view stream, void* arg) {
    std::cout << "Deallocating " << size << " bytes" << std::endl;
    auto base_mr = *static_cast<rmm::device_async_resource_ref*>(arg);
    base_mr.deallocate_async(ptr, size, stream);
  };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, &base_mr, &base_mr);
  auto const size = std::size_t{10_MiB};
  auto* ptr       = mr.allocate(size);
  mr.deallocate(ptr, size);

  auto output = testing::internal::GetCapturedStdout();
  auto expect = std::string("Allocating ") + std::to_string(size) + " bytes\nDeallocating " +
                std::to_string(size) + " bytes\n";
  ASSERT_EQ(expect, output);
}

}  // namespace
}  // namespace rmm::test
