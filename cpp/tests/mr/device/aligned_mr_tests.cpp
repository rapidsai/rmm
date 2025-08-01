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

#include "../../mock_resource.hpp"

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

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
    EXPECT_EQ(mr.allocate(size, stream), pointer);
    mr.deallocate(pointer, size, stream);
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
    EXPECT_EQ(mr.allocate(size, stream), pointer);
    mr.deallocate(pointer, size, stream);
  }

  {
    auto const size{65528};
    void* const pointer1 = int_to_address(456);
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer1));
    EXPECT_CALL(mock, do_deallocate(pointer1, size, stream)).Times(1);
    EXPECT_EQ(mr.allocate(size, stream), pointer1);
    mr.deallocate(pointer1, size, stream);
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
    EXPECT_EQ(mr.allocate(size, stream), pointer);
    mr.deallocate(pointer, size, stream);
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
    EXPECT_EQ(mr.allocate(size, stream), expected_pointer);
    mr.deallocate(expected_pointer, size, stream);
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
    EXPECT_EQ(mr.allocate(size1, stream), expected_pointer1);
    EXPECT_EQ(mr.allocate(size2, stream), expected_pointer2);
    EXPECT_EQ(mr.allocate(size3, stream), expected_pointer3);
    mr.deallocate(expected_pointer1, size1, stream);
    mr.deallocate(expected_pointer2, size2, stream);
    mr.deallocate(expected_pointer3, size3, stream);
  }
}

TEST(AlignedTest, AlignRealPointer)
{
  auto const alignment{4096};
  auto const threshold{65536};
  aligned_real mr{rmm::mr::get_current_device_resource_ref(), alignment, threshold};
  void* alloc = mr.allocate(threshold);
  EXPECT_TRUE(rmm::is_pointer_aligned(alloc, alignment));
  mr.deallocate(alloc, threshold);
}

TEST(AlignedTest, SmallAlignmentsBumpedTo256Bytes)
{
  // Test various small alignments
  for (auto requested_alignment : {32UL, 64UL, 128UL}) {
    aligned_real mr{rmm::mr::get_current_device_resource_ref(), requested_alignment};

    void* ptr = mr.allocate(requested_alignment);

    // Even though we requested smaller alignment, pointer should be 256-byte
    // aligned for CUDA requirements
    EXPECT_TRUE(rmm::is_pointer_aligned(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT));
    // And also aligned to the originally requested alignment
    EXPECT_TRUE(rmm::is_pointer_aligned(ptr, requested_alignment));

    mr.deallocate(ptr, requested_alignment);
  }
}

}  // namespace
}  // namespace rmm::test
