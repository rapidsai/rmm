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
  EXPECT_THROW(construct_alignment(&mock, 768), rmm::logic_error);
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
  void* pointer = reinterpret_cast<void*>(123);
  // device_memory_resource aligns to 8.
  EXPECT_CALL(mock, do_allocate(8, stream)).WillOnce(Return(pointer));
  EXPECT_CALL(mock, do_deallocate(pointer, 8, stream)).Times(1);
  EXPECT_EQ(mr.allocate(5, stream), pointer);
  mr.deallocate(pointer, 5, stream);
}

TEST(AlignedTest, BelowAlignmentThresholdPassthrough)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* pointer = reinterpret_cast<void*>(123);
  // device_memory_resource aligns to 8.
  EXPECT_CALL(mock, do_allocate(8, stream)).WillOnce(Return(pointer));
  EXPECT_CALL(mock, do_deallocate(pointer, 8, stream)).Times(1);
  EXPECT_EQ(mr.allocate(3, stream), pointer);
  mr.deallocate(pointer, 3, stream);

  void* pointer1 = reinterpret_cast<void*>(456);
  EXPECT_CALL(mock, do_allocate(65528, stream)).WillOnce(Return(pointer1));
  EXPECT_CALL(mock, do_deallocate(pointer1, 65528, stream)).Times(1);
  EXPECT_EQ(mr.allocate(65528, stream), pointer1);
  mr.deallocate(pointer1, 65528, stream);
}

TEST(AlignedTest, UpstreamAddressAlreadyAligned)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* pointer = reinterpret_cast<void*>(4096);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillOnce(Return(pointer));
  EXPECT_CALL(mock, do_deallocate(pointer, 69376, stream)).Times(1);

  EXPECT_EQ(mr.allocate(65536, stream), pointer);
  mr.deallocate(pointer, 65536, stream);
}

TEST(AlignedTest, AlignUpstreamAddress)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* pointer = reinterpret_cast<void*>(256);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillOnce(Return(pointer));
  EXPECT_CALL(mock, do_deallocate(pointer, 69376, stream)).Times(1);

  void* expected_pointer = reinterpret_cast<void*>(4096);
  EXPECT_EQ(mr.allocate(65536, stream), expected_pointer);
  mr.deallocate(expected_pointer, 65536, stream);
}

TEST(AlignedTest, AlignMultiple)
{
  mock_resource mock;
  aligned_adaptor mr{&mock, 4096, 65536};

  cuda_stream_view stream;
  void* pointer  = reinterpret_cast<void*>(256);
  void* pointer1 = reinterpret_cast<void*>(131584);
  void* pointer2 = reinterpret_cast<void*>(263168);
  EXPECT_CALL(mock, do_allocate(69376, stream)).WillOnce(Return(pointer));
  EXPECT_CALL(mock, do_allocate(77568, stream)).WillOnce(Return(pointer1));
  EXPECT_CALL(mock, do_allocate(81664, stream)).WillOnce(Return(pointer2));
  EXPECT_CALL(mock, do_deallocate(pointer, 69376, stream)).Times(1);
  EXPECT_CALL(mock, do_deallocate(pointer1, 77568, stream)).Times(1);
  EXPECT_CALL(mock, do_deallocate(pointer2, 81664, stream)).Times(1);

  void* expected_pointer  = reinterpret_cast<void*>(4096);
  void* expected_pointer1 = reinterpret_cast<void*>(135168);
  void* expected_pointer2 = reinterpret_cast<void*>(266240);
  EXPECT_EQ(mr.allocate(65536, stream), expected_pointer);
  EXPECT_EQ(mr.allocate(73728, stream), expected_pointer1);
  EXPECT_EQ(mr.allocate(77800, stream), expected_pointer2);
  mr.deallocate(expected_pointer1, 73728, stream);
  mr.deallocate(expected_pointer, 65536, stream);
  mr.deallocate(expected_pointer2, 77800, stream);
}

}  // namespace
}  // namespace rmm::test
