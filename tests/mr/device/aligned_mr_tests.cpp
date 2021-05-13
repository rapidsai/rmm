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

using aligned_adaptor = rmm::mr::aligned_resource_adaptor<mock_resource>;

TEST(AlignedTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { aligned_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(AlignedTest, ThrowOnInvalidAllocationAlignment)
{
  mock_resource mock;
  auto construct_alignment = [](auto* r, std::size_t a) { aligned_adaptor mr{r, a}; };
  EXPECT_THROW(construct_alignment(&mock, 255), rmm::logic_error);
  EXPECT_NO_THROW(construct_alignment(&mock, 256));
  EXPECT_THROW(construct_alignment(&mock, 257), rmm::logic_error);
}

TEST(AlignedTest, ThrowOnInvalidAlignmentThreshold)
{
  mock_resource mock;
  auto construct_threshold = [](auto* r, std::size_t t) { aligned_adaptor mr{r, 4096, t}; };
  EXPECT_THROW(construct_threshold(&mock, 65535), rmm::logic_error);
  EXPECT_NO_THROW(construct_threshold(&mock, 65536));
  EXPECT_THROW(construct_threshold(&mock, 65537), rmm::logic_error);
}

TEST(AlignedTest, SupportsStreams)
{
  mock_resource mock;
  aligned_adaptor mr{&mock};

  EXPECT_CALL(mock, supports_streams()).WillOnce(Return(true));
  EXPECT_TRUE(mr.supports_streams());

  EXPECT_CALL(mock, supports_streams()).WillOnce(Return(false));
  EXPECT_FALSE(mr.supports_streams());
}

TEST(AlignedTest, SupportsGetMemInfo)
{
  mock_resource mock;
  aligned_adaptor mr{&mock};

  EXPECT_CALL(mock, supports_get_mem_info()).WillOnce(Return(true));
  EXPECT_TRUE(mr.supports_get_mem_info());

  EXPECT_CALL(mock, supports_get_mem_info()).WillOnce(Return(false));
  EXPECT_FALSE(mr.supports_get_mem_info());
}

TEST(AlignedTest, DefaultAllocationAlignmentPassthrough)
{
  mock_resource mock;
  aligned_adaptor mr{&mock};

  cuda_stream_view stream;
  void* address = reinterpret_cast<void*>(123);
  // device_memory_resource aligns to 8.
  EXPECT_CALL(mock, do_allocate(8, stream)).WillOnce(Return(address));
  EXPECT_CALL(mock, do_deallocate(address, 8, stream)).Times(1);
  EXPECT_EQ(mr.allocate(5, stream), address);
  mr.deallocate(address, 5, stream);
}

TEST(AlignedTest, BelowAlignmentThresholdPassthrough)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* address = reinterpret_cast<void*>(123);
  // device_memory_resource aligns to 8.
  EXPECT_CALL(mock, do_allocate(8, stream)).WillRepeatedly(Return(address));
  EXPECT_CALL(mock, do_deallocate(address, 8, stream)).Times(1);
  EXPECT_EQ(mr.allocate(3, stream), address);
  mr.deallocate(address, 3, stream);

  void* address1 = reinterpret_cast<void*>(456);
  EXPECT_CALL(mock, do_allocate(65528, stream)).WillOnce(Return(address1));
  EXPECT_CALL(mock, do_deallocate(address1, 65528, stream)).Times(1);
  EXPECT_EQ(mr.allocate(65528, stream), address1);
  mr.deallocate(address1, 65528, stream);
}

TEST(AlignedTest, UpstreamAddressAlreadyAligned)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* address = reinterpret_cast<void*>(4096);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillRepeatedly(Return(address));
  void* tail_address = reinterpret_cast<void*>(69632);
  EXPECT_CALL(mock, do_deallocate(tail_address, 3840, stream)).Times(1);
  EXPECT_EQ(mr.allocate(65536, stream), address);

  EXPECT_CALL(mock, do_deallocate(address, 65536, stream)).Times(1);
  mr.deallocate(address, 65536, stream);
}

TEST(AlignedTest, ReturnHeadOnly)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* address = reinterpret_cast<void*>(256);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillRepeatedly(Return(address));
  EXPECT_CALL(mock, do_deallocate(address, 3840, stream)).Times(1);
  void* expected_address = reinterpret_cast<void*>(4096);
  EXPECT_EQ(mr.allocate(65536, stream), expected_address);

  EXPECT_CALL(mock, do_deallocate(expected_address, 65536, stream)).Times(1);
  mr.deallocate(expected_address, 65536, stream);
}

TEST(AlignedTest, ReturnBothHeadAndTail)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* address = reinterpret_cast<void*>(768);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillRepeatedly(Return(address));
  EXPECT_CALL(mock, do_deallocate(address, 3328, stream)).Times(1);
  void* tail_address = reinterpret_cast<void*>(69632);
  EXPECT_CALL(mock, do_deallocate(tail_address, 512, stream)).Times(1);
  void* expected_address = reinterpret_cast<void*>(4096);
  EXPECT_EQ(mr.allocate(65536, stream), expected_address);

  EXPECT_CALL(mock, do_deallocate(expected_address, 65536, stream)).Times(1);
  mr.deallocate(expected_address, 65536, stream);
}

}  // namespace
}  // namespace rmm::test
