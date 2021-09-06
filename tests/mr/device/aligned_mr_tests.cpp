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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using ::testing::Return;

class mock_resource : public rmm::mr::device_memory_resource {
 public:
  MOCK_METHOD(bool, supports_streams, (), (const, override, noexcept));
  MOCK_METHOD(bool, supports_get_mem_info, (), (const, override, noexcept));
  MOCK_METHOD(void*, do_allocate, (std::size_t, cuda_stream_view), (override));
  MOCK_METHOD(void, do_deallocate, (void*, std::size_t, cuda_stream_view), (override));
  using size_pair = std::pair<std::size_t, std::size_t>;
  MOCK_METHOD(size_pair, do_get_mem_info, (cuda_stream_view), (const, override));
};

using aligned_mock = rmm::mr::aligned_resource_adaptor<mock_resource>;
using aligned_real = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>;

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

TEST(AlignedTest, SupportsStreams)
{
  mock_resource mock;
  aligned_mock mr{&mock};

  EXPECT_CALL(mock, supports_streams()).WillOnce(Return(true));
  EXPECT_TRUE(mr.supports_streams());

  EXPECT_CALL(mock, supports_streams()).WillOnce(Return(false));
  EXPECT_FALSE(mr.supports_streams());
}

TEST(AlignedTest, SupportsGetMemInfo)
{
  mock_resource mock;
  aligned_mock mr{&mock};

  EXPECT_CALL(mock, supports_get_mem_info()).WillOnce(Return(true));
  EXPECT_TRUE(mr.supports_get_mem_info());

  EXPECT_CALL(mock, supports_get_mem_info()).WillOnce(Return(false));
  EXPECT_FALSE(mr.supports_get_mem_info());
}

TEST(AlignedTest, DefaultAllocationAlignmentPassthrough)
{
  mock_resource mock;
  aligned_mock mr{&mock};

  cuda_stream_view stream;
  auto const unaligned_address{123};
  void* const pointer = reinterpret_cast<void*>(unaligned_address);
  // device_memory_resource aligns to 8.
  {
    auto const size{8};
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
  auto const unaligned_address1{123};
  void* const pointer = reinterpret_cast<void*>(unaligned_address1);
  // device_memory_resource aligns to 8.
  {
    auto const size{8};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    auto const size{3};
    EXPECT_EQ(mr.allocate(size, stream), pointer);
    mr.deallocate(pointer, size, stream);
  }

  {
    auto const unaligned_address2{456};
    auto const size{65528};
    void* const pointer1 = reinterpret_cast<void*>(unaligned_address2);
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
  auto const aligned_address{4096};
  void* const pointer = reinterpret_cast<void*>(aligned_address);

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
    auto const address{256};
    void* const pointer = reinterpret_cast<void*>(address);
    auto const size{69376};
    EXPECT_CALL(mock, do_allocate(size, stream)).WillOnce(Return(pointer));
    EXPECT_CALL(mock, do_deallocate(pointer, size, stream)).Times(1);
  }

  {
    auto const address{4096};
    void* const expected_pointer = reinterpret_cast<void*>(address);
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
    auto const address1{256};
    auto const address2{131584};
    auto const address3{263168};
    void* const pointer1 = reinterpret_cast<void*>(address1);
    void* const pointer2 = reinterpret_cast<void*>(address2);
    void* const pointer3 = reinterpret_cast<void*>(address3);
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
    auto const expected_address1{4096};
    auto const expected_address2{135168};
    auto const expected_address3{266240};
    void* const expected_pointer1 = reinterpret_cast<void*>(expected_address1);
    void* const expected_pointer2 = reinterpret_cast<void*>(expected_address2);
    void* const expected_pointer3 = reinterpret_cast<void*>(expected_address3);
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
  aligned_real mr{rmm::mr::get_current_device_resource(), alignment, threshold};
  void* alloc        = mr.allocate(threshold);
  auto const address = reinterpret_cast<std::size_t>(alloc);
  EXPECT_TRUE(address % alignment == 0);
  mr.deallocate(alloc, threshold);
}

}  // namespace
}  // namespace rmm::test
