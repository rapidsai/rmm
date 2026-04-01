/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../mock_resource.hpp"
#include "delayed_memory_resource.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>

namespace rmm::test {
namespace {

using ::testing::Return;

using aligned_mock = rmm::mr::aligned_resource_adaptor<mock_resource>;
using aligned_real = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>;

void* int_to_address(std::size_t val)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, performance-no-int-to-ptr)
  return reinterpret_cast<void*>(val);
}

struct allocation_size : public ::testing::TestWithParam<std::size_t> {};

INSTANTIATE_TEST_SUITE_P(AlignedTest, allocation_size, ::testing::Values(0, 256));

TEST_P(allocation_size, MultiThreaded)
{
  const std::size_t allocation_size = GetParam();
  auto upstream                     = rmm::mr::cuda_memory_resource{};
  auto delayed = delayed_memory_resource(upstream, std::chrono::milliseconds{300});
  auto mr      = rmm::mr::aligned_resource_adaptor<delayed_memory_resource>(delayed);
  auto stream  = rmm::cuda_stream{};
  // Provoke interleaving to test that aligned allocations are updated with correct ordering
  // relative to upstream deallocate. The delayed memory resource frees the pointer upstream
  // immediately then sleeps, simulating the window where the address is available for reuse
  // but the adaptor hasn't updated its counters yet.
  //
  // Thread-0             Thread-1
  // alloc
  //                      alloc
  //                      dealloc-start
  // dealloc-start
  //                      dealloc-end
  // dealloc-end
  //
  // After both threads complete, the counters must reflect zero outstanding allocations.
  std::vector<std::thread> threads;
  for (int i = 0; i < 2; i++) {
    threads.emplace_back([&, i = i]() {
      void* ptr{nullptr};
      if (i != 0) { std::this_thread::sleep_for(std::chrono::milliseconds{100}); }
      EXPECT_NO_THROW(ptr = mr.allocate(stream, allocation_size));
      if (allocation_size != 0) {
        EXPECT_NE(ptr, nullptr);
      } else {
        EXPECT_EQ(ptr, nullptr);
      }
      if (i == 0) { std::this_thread::sleep_for(std::chrono::milliseconds{100}); }
      mr.deallocate(stream, ptr, allocation_size);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST(AlignedTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { aligned_mock mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(AlignedTest, ThrowOnInvalidAllocationAlignment)
{
  mock_resource mock;
  auto construct_alignment = [](auto* memres, std::size_t align) {
    aligned_mock mr{memres, align};
  };
  EXPECT_THROW(construct_alignment(&mock, 255), rmm::logic_error);
  EXPECT_NO_THROW(construct_alignment(&mock, 256));
  EXPECT_THROW(construct_alignment(&mock, 768), rmm::logic_error);
}

TEST(AlignedTest, SupportsGetMemInfo)
{
  mock_resource mock;
  aligned_mock mr{mock};
}

TEST(AlignedTest, DefaultAllocationAlignmentPassthrough)
{
  mock_resource mock;
  aligned_mock mr{mock};

  cuda_stream_view stream;
  void* const pointer = int_to_address(123);

  {
    auto const size{5};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    auto const size{5};
    EXPECT_EQ(mr.allocate(stream, size), pointer);
    mr.deallocate(stream, pointer, size);
  }
}

TEST(AlignedTest, BelowAlignmentThresholdPassthrough)
{
  mock_resource mock;
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_mock mr{&mock, alignment, threshold};

  cuda_stream_view stream;
  void* const pointer = int_to_address(123);
  {
    auto const size{3};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    auto const size{3};
    EXPECT_EQ(mr.allocate(stream, size), pointer);
    mr.deallocate(stream, pointer, size);
  }

  {
    auto const size{65528};
    void* const pointer1 = int_to_address(456);
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer1));
    EXPECT_CALL(mock, do_deallocate(pointer1, size, stream)).Times(1);
    EXPECT_EQ(mr.allocate(stream, size), pointer1);
    mr.deallocate(stream, pointer1, size);
  }
}

TEST(AlignedTest, UpstreamAddressAlreadyAligned)
{
  mock_resource mock;
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_mock mr{&mock, alignment, threshold};

  cuda_stream_view stream;
  void* const pointer = int_to_address(4096);

  {
    auto const size{69376};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    auto const size{65536};
    EXPECT_EQ(mr.allocate(stream, size), pointer);
    mr.deallocate(stream, pointer, size);
  }
}

TEST(AlignedTest, AlignUpstreamAddress)
{
  mock_resource mock;
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_mock mr{&mock, alignment, threshold};

  cuda_stream_view stream;
  {
    void* const pointer = int_to_address(256);
    auto const size{69376};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    void* const expected_pointer = int_to_address(4096);
    auto const size{65536};
    EXPECT_EQ(mr.allocate(stream, size), expected_pointer);
    mr.deallocate(stream, expected_pointer, size);
  }
}

TEST(AlignedTest, AlignMultiple)
{
  mock_resource mock;
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_mock mr{&mock, alignment, threshold};

  cuda_stream_view stream;

  {
    void* const pointer1 = int_to_address(256);
    void* const pointer2 = int_to_address(131584);
    void* const pointer3 = int_to_address(263168);
    auto const size1{69376};
    auto const size2{77568};
    auto const size3{81664};
    EXPECT_CALL(mock, do_allocate(size1, stream)).WillOnce(Return(pointer1));
    EXPECT_CALL(mock, do_allocate(size2, stream)).WillOnce(Return(pointer2));
    EXPECT_CALL(mock, do_allocate(size3, stream)).WillOnce(Return(pointer3));
    EXPECT_CALL(mock, do_deallocate(pointer1, size1, stream)).Times(1);
    EXPECT_CALL(mock, do_deallocate(pointer2, size2, stream)).Times(1);
    EXPECT_CALL(mock, do_deallocate(pointer3, size3, stream)).Times(1);
  }

  {
    void* const expected_pointer1 = int_to_address(4096);
    void* const expected_pointer2 = int_to_address(135168);
    void* const expected_pointer3 = int_to_address(266240);
    auto const size1{65536};
    auto const size2{73728};
    auto const size3{77800};
    EXPECT_EQ(mr.allocate(stream, size1), expected_pointer1);
    EXPECT_EQ(mr.allocate(stream, size2), expected_pointer2);
    EXPECT_EQ(mr.allocate(stream, size3), expected_pointer3);
    mr.deallocate(stream, expected_pointer1, size1);
    mr.deallocate(stream, expected_pointer2, size2);
    mr.deallocate(stream, expected_pointer3, size3);
  }
}

TEST(AlignedTest, AlignRealPointer)
{
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_real mr{rmm::mr::get_current_device_resource_ref(), alignment, threshold};
  void* alloc = mr.allocate_sync(threshold);
  EXPECT_TRUE(rmm::is_pointer_aligned(alloc, alignment));
  mr.deallocate_sync(alloc, threshold);
}

TEST(AlignedTest, SmallAlignmentsBumpedTo256Bytes)
{
  // Test various small alignments
  for (auto requested_alignment : {32UL, 64UL, 128UL}) {
    aligned_real mr{rmm::mr::get_current_device_resource_ref(), requested_alignment};

    void* ptr = mr.allocate_sync(requested_alignment);

    // Even though we requested smaller alignment, pointer should be 256-byte
    // aligned for CUDA requirements
    EXPECT_TRUE(rmm::is_pointer_aligned(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT));
    // And also aligned to the originally requested alignment
    EXPECT_TRUE(rmm::is_pointer_aligned(ptr, requested_alignment));

    mr.deallocate_sync(ptr, requested_alignment);
  }
}

}  // namespace
}  // namespace rmm::test
