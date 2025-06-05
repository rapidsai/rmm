/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace rmm::test {
namespace {

template <typename ExceptionType = rmm::bad_alloc>
using failure_callback_adaptor =
  rmm::mr::failure_callback_resource_adaptor<rmm::mr::device_memory_resource, ExceptionType>;

bool failure_handler(std::size_t /*bytes*/, void* arg)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  bool& retried = *reinterpret_cast<bool*>(arg);
  if (!retried) {
    retried = true;
    return true;  // First time we request an allocation retry
  }
  return false;  // Second time we let the adaptor throw std::bad_alloc
}

TEST(FailureCallbackTest, RetryAllocationOnce)
{
  bool retried{false};
  failure_callback_adaptor<> mr{
    rmm::mr::get_current_device_resource_ref(), failure_handler, &retried};
  EXPECT_EQ(retried, false);
  EXPECT_THROW(mr.allocate(512_GiB), std::bad_alloc);
  EXPECT_EQ(retried, true);
}

template <typename ExceptionType>
class always_throw_memory_resource final : public mr::device_memory_resource {
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    throw ExceptionType{"foo"};
  }
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override {};
};

TEST(FailureCallbackTest, DifferentExceptionTypes)
{
  always_throw_memory_resource<rmm::bad_alloc> bad_alloc_mr;
  always_throw_memory_resource<rmm::out_of_memory> oom_mr;

  EXPECT_THROW(bad_alloc_mr.allocate(1_MiB), rmm::bad_alloc);
  EXPECT_THROW(oom_mr.allocate(1_MiB), rmm::out_of_memory);

  // Wrap a bad_alloc-catching callback adaptor around an MR that always throws bad_alloc:
  // Should retry once and then re-throw bad_alloc
  {
    bool retried{false};
    failure_callback_adaptor<rmm::bad_alloc> bad_alloc_callback_mr{
      &bad_alloc_mr, failure_handler, &retried};

    EXPECT_EQ(retried, false);
    EXPECT_THROW(bad_alloc_callback_mr.allocate(1_MiB), rmm::bad_alloc);
    EXPECT_EQ(retried, true);
  }

  // Wrap a out_of_memory-catching callback adaptor around an MR that always throws out_of_memory:
  // Should retry once and then re-throw out_of_memory
  {
    bool retried{false};

    failure_callback_adaptor<rmm::out_of_memory> oom_callback_mr{
      &oom_mr, failure_handler, &retried};
    EXPECT_EQ(retried, false);
    EXPECT_THROW(oom_callback_mr.allocate(1_MiB), rmm::out_of_memory);
    EXPECT_EQ(retried, true);
  }

  // Wrap a out_of_memory-catching callback adaptor around an MR that always throws bad_alloc:
  // Should not catch the bad_alloc exception
  {
    bool retried{false};

    failure_callback_adaptor<rmm::out_of_memory> oom_callback_mr{
      &bad_alloc_mr, failure_handler, &retried};
    EXPECT_EQ(retried, false);
    EXPECT_THROW(oom_callback_mr.allocate(1_MiB), rmm::bad_alloc);  // bad_alloc passes through
    EXPECT_EQ(retried, false);  // Does not catch / retry on anything except OOM
  }
}

}  // namespace
}  // namespace rmm::test
