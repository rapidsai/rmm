/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/failure_alternate_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <unordered_set>

namespace rmm::test {
namespace {

template <typename ExceptionType>
struct throw_at_limit_resource final : public mr::device_memory_resource {
  throw_at_limit_resource(std::size_t limit) : limit{limit} {}

  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    if (bytes > limit) { throw ExceptionType{"foo"}; }
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&ptr, bytes));
    allocs.insert(ptr);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(ptr));
    allocs.erase(ptr);
  }

  [[nodiscard]] bool do_is_equal(mr::device_memory_resource const& other) const noexcept override
  {
    return this == &other;
  }

  const std::size_t limit;
  std::unordered_set<void*> allocs{};
};

TEST(FailureAlternateTest, TrackBothUpstreams)
{
  throw_at_limit_resource<rmm::out_of_memory> primary_mr{100};
  throw_at_limit_resource<rmm::out_of_memory> alternate_mr{1000};
  rmm::mr::failure_alternate_resource_adaptor<rmm::out_of_memory> mr{primary_mr, alternate_mr};

  // Check that a small allocation goes to the primary resource
  {
    void* a1 = mr.allocate(10);
    EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{{a1}});
    EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
    mr.deallocate(a1, 10);
    EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
    EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
  }

  // Check that a large allocation goes to the alternate resource
  {
    void* a1 = mr.allocate(200);
    EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
    EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{a1});
    mr.deallocate(a1, 200);
    EXPECT_EQ(primary_mr.allocs, std::unordered_set<void*>{});
    EXPECT_EQ(alternate_mr.allocs, std::unordered_set<void*>{});
  }

  // Check that the exceptions raised by the alternate isn't caught
  EXPECT_THROW(mr.allocate(2000), rmm::out_of_memory);
}

TEST(FailureAlternateTest, DifferentExceptionTypes)
{
  throw_at_limit_resource<std::invalid_argument> primary_mr{100};
  throw_at_limit_resource<rmm::out_of_memory> alternate_mr{1000};
  rmm::mr::failure_alternate_resource_adaptor<rmm::out_of_memory> mr{primary_mr, alternate_mr};

  // Check that only `rmm::out_of_memory` exceptions are caught
  EXPECT_THROW(mr.allocate(200), std::invalid_argument);
}

}  // namespace
}  // namespace rmm::test
