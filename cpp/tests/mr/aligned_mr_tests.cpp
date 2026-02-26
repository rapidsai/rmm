/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../mock_resource.hpp"

#include <rmm/aligned.hpp>
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

using aligned_adaptor = rmm::mr::aligned_resource_adaptor;

void* int_to_address(std::size_t val)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, performance-no-int-to-ptr)
  return reinterpret_cast<void*>(val);
}

TEST(AlignedTest, ThrowOnInvalidAllocationAlignment)
{
  mock_resource mock;
  auto construct_alignment = [](mock_resource& memres, std::size_t align) {
    aligned_adaptor mr{memres, align};
  };
  EXPECT_THROW(construct_alignment(mock, 255), rmm::logic_error);
  EXPECT_NO_THROW(construct_alignment(mock, 256));
  EXPECT_THROW(construct_alignment(mock, 768), rmm::logic_error);
}

TEST(AlignedTest, SupportsGetMemInfo)
{
  mock_resource mock;
  aligned_adaptor mr{mock};
}

TEST(AlignedTest, DefaultAllocationAlignmentPassthrough)
{
  mock_resource mock;
  aligned_adaptor mr{mock};

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
  aligned_adaptor mr{mock, alignment, threshold};

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
  aligned_adaptor mr{mock, alignment, threshold};

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
  aligned_adaptor mr{mock, alignment, threshold};

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
  aligned_adaptor mr{mock, alignment, threshold};

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
  aligned_adaptor mr{rmm::mr::get_current_device_resource_ref(), alignment, threshold};
  void* alloc = mr.allocate_sync(threshold);
  EXPECT_TRUE(rmm::is_pointer_aligned(alloc, alignment));
  mr.deallocate_sync(alloc, threshold);
}

TEST(AlignedTest, SmallAlignmentsBumpedTo256Bytes)
{
  // Test various small alignments
  for (auto requested_alignment : {32UL, 64UL, 128UL}) {
    aligned_adaptor mr{rmm::mr::get_current_device_resource_ref(), requested_alignment};

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
