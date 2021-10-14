/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cstddef>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/oom_callback_resource_adaptor.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using oom_callback_adaptor =
  rmm::mr::oom_callback_resource_adaptor<rmm::mr::device_memory_resource>;

typedef struct {
  bool retried;
} cb_arg;

bool oom_handler(std::size_t bytes, void* arg)
{
  cb_arg* a = reinterpret_cast<cb_arg*>(arg);
  if (!a->retried) {
    a->retried = true;
    return true;  // First time we request an allocation retry
  } else {
    return false;  // Second time we let the adaptor throw std::bad_alloc
  }
}

TEST(OOMCallbackTest, RetryAllocationOnce)
{
  cb_arg arg{false};
  oom_callback_adaptor mr{rmm::mr::get_current_device_resource(), oom_handler, &arg};
  rmm::mr::set_current_device_resource(&mr);
  EXPECT_EQ(arg.retried, false);
  EXPECT_THROW(mr.allocate(100_GiB), std::bad_alloc);
  EXPECT_EQ(arg.retried, true);
}

}  // namespace
}  // namespace rmm::test
