/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <gtest/gtest.h>

#include <cstddef>

namespace rmm::test {
namespace {

template <typename ExceptionType = rmm::bad_alloc>
using failure_callback_adaptor = rmm::mr::failure_callback_resource_adaptor<ExceptionType>;

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

template <typename ExceptionType>
class always_throw_memory_resource final {
 public:
  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t /*bytes*/,
                 std::size_t /*alignment*/ = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    throw ExceptionType{"foo"};
  }
  void deallocate(cuda::stream_ref /*stream*/,
                  void* /*ptr*/,
                  std::size_t /*bytes*/,
                  std::size_t /*alignment*/ = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {};

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(always_throw_memory_resource const&) const noexcept { return true; }
  bool operator!=(always_throw_memory_resource const&) const noexcept { return false; }
  constexpr friend void get_property(always_throw_memory_resource const&,
                                     cuda::mr::device_accessible) noexcept
  {
  }
};

TEST(FailureCallbackTest, RetryAllocationOnce)
{
  always_throw_memory_resource<rmm::bad_alloc> throwing_mr;
  bool retried{false};
  failure_callback_adaptor<> mr{throwing_mr, failure_handler, &retried};
  EXPECT_EQ(retried, false);
  EXPECT_THROW((void)mr.allocate_sync(1_MiB), rmm::bad_alloc);
  EXPECT_EQ(retried, true);
}

TEST(FailureCallbackTest, DifferentExceptionTypes)
{
  always_throw_memory_resource<rmm::bad_alloc> bad_alloc_mr;
  always_throw_memory_resource<rmm::out_of_memory> oom_mr;

  EXPECT_THROW((void)bad_alloc_mr.allocate_sync(1_MiB), rmm::bad_alloc);
  EXPECT_THROW((void)oom_mr.allocate_sync(1_MiB), rmm::out_of_memory);

  // Wrap a bad_alloc-catching callback adaptor around an MR that always throws bad_alloc:
  // Should retry once and then re-throw bad_alloc
  {
    bool retried{false};
    failure_callback_adaptor<rmm::bad_alloc> bad_alloc_callback_mr{
      bad_alloc_mr, failure_handler, &retried};

    EXPECT_EQ(retried, false);
    EXPECT_THROW((void)bad_alloc_callback_mr.allocate_sync(1_MiB), rmm::bad_alloc);
    EXPECT_EQ(retried, true);
  }

  // Wrap a out_of_memory-catching callback adaptor around an MR that always throws out_of_memory:
  // Should retry once and then re-throw out_of_memory
  {
    bool retried{false};

    failure_callback_adaptor<rmm::out_of_memory> oom_callback_mr{oom_mr, failure_handler, &retried};
    EXPECT_EQ(retried, false);
    EXPECT_THROW((void)oom_callback_mr.allocate_sync(1_MiB), rmm::out_of_memory);
    EXPECT_EQ(retried, true);
  }

  // Wrap a out_of_memory-catching callback adaptor around an MR that always throws bad_alloc:
  // Should not catch the bad_alloc exception
  {
    bool retried{false};

    failure_callback_adaptor<rmm::out_of_memory> oom_callback_mr{
      bad_alloc_mr, failure_handler, &retried};
    EXPECT_EQ(retried, false);
    EXPECT_THROW((void)oom_callback_mr.allocate_sync(1_MiB),
                 rmm::bad_alloc);  // bad_alloc passes through
    EXPECT_EQ(retried, false);     // Does not catch / retry on anything except OOM
  }
}

}  // namespace
}  // namespace rmm::test
