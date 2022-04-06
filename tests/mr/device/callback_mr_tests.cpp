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

#include "../../byte_literals.hpp"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"

#include <cstddef>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/callback_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

TEST(CallbackTest, PassThroughTest)
{
  auto base_mr           = rmm::mr::get_current_device_resource();
  auto allocate_callback = [&base_mr](std::size_t size, void* arg, cuda_stream_view stream) {
    return base_mr->allocate(size, stream);
  };
  auto deallocate_callback = [&base_mr](
                               void* ptr, std::size_t size, void* arg, cuda_stream_view stream) {
    base_mr->deallocate(ptr, size, stream);
  };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, nullptr, nullptr);
  auto ptr = mr.allocate(10_MiB);
  mr.deallocate(ptr, 10_MiB);
}

TEST(CallbackTest, LoggingTest)
{
  testing::internal::CaptureStdout();
  
  auto base_mr           = rmm::mr::get_current_device_resource();
  auto allocate_callback = [&base_mr](std::size_t size, void* arg, cuda_stream_view stream) {
    std::cout << "Allocating " << size << " bytes" << std::endl;
    return base_mr->allocate(size, stream);
  };
  auto deallocate_callback = [&base_mr](
                               void* ptr, std::size_t size, void* arg, cuda_stream_view stream) {
    std::cout << "Deallocating " << size << " bytes" << std::endl;    
    base_mr->deallocate(ptr, size, stream);
  };
  auto mr =
    rmm::mr::callback_memory_resource(allocate_callback, deallocate_callback, nullptr, nullptr);
  auto ptr = mr.allocate(10_MiB);
  mr.deallocate(ptr, 10_MiB);

  std::string output = testing::internal::GetCapturedStdout();
  std::string expect {"Allocating " + std::to_string(10_MiB) + " bytes" + "\n" + "Deallocating " + std::to_string(10_MiB) + " bytes\n"};
  ASSERT_EQ(expect, output);
}
  
}  // namespace
}  // namespace rmm::test
