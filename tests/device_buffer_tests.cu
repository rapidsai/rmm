/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <cstddef>
#include <random>

template <typename MemoryResourceType>
struct DeviceBufferTest : public ::testing::Test {
  rmm::cuda_stream stream{};
  std::size_t size{};
  MemoryResourceType mr{};

  DeviceBufferTest()
  {
    std::default_random_engine generator;
    std::uniform_int_distribution<std::size_t> distribution(1000, 100000);
    size = distribution(generator);
  }
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource, rmm::mr::managed_memory_resource>;

TYPED_TEST_CASE(DeviceBufferTest, resources);

TYPED_TEST(DeviceBufferTest, DefaultMemoryResource)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_view{});
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(rmm::mr::get_current_device_resource(), buff.memory_resource());
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());
}

TYPED_TEST(DeviceBufferTest, DefaultMemoryResourceStream)
{
  rmm::device_buffer buff(this->size, this->stream);
  this->stream.synchronize();
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(rmm::mr::get_current_device_resource(), buff.memory_resource());
  EXPECT_EQ(this->stream, buff.stream());
}

TYPED_TEST(DeviceBufferTest, ExplicitMemoryResource)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_view{}, &this->mr);
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(&this->mr, buff.memory_resource());
  EXPECT_TRUE(this->mr.is_equal(*buff.memory_resource()));
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());
}

TYPED_TEST(DeviceBufferTest, ExplicitMemoryResourceStream)
{
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  this->stream.synchronize();
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(&this->mr, buff.memory_resource());
  EXPECT_TRUE(this->mr.is_equal(*buff.memory_resource()));
  EXPECT_EQ(this->stream, buff.stream());
}

TYPED_TEST(DeviceBufferTest, CopyFromRawDevicePointer)
{
  void* device_memory{nullptr};
  EXPECT_EQ(cudaSuccess, cudaMalloc(&device_memory, this->size));
  rmm::device_buffer buff(device_memory, this->size, rmm::cuda_stream_view{});
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(rmm::mr::get_current_device_resource(), buff.memory_resource());
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());

  // TODO check for equality between the contents of the two allocations
  buff.stream().synchronize();
  EXPECT_EQ(cudaSuccess, cudaFree(device_memory));
}

TYPED_TEST(DeviceBufferTest, CopyFromRawHostPointer)
{
  std::vector<uint8_t> host_data(this->size);
  rmm::device_buffer buff(
    static_cast<void*>(host_data.data()), this->size, rmm::cuda_stream_view{});
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(this->size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());
  EXPECT_EQ(rmm::mr::get_current_device_resource(), buff.memory_resource());
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());
  buff.stream().synchronize();
  // TODO check for equality between the contents of the two allocations
}

TYPED_TEST(DeviceBufferTest, CopyFromNullptr)
{
  // can  copy from a nullptr only if size == 0
  rmm::device_buffer buff(nullptr, 0, rmm::cuda_stream_view{});
  EXPECT_EQ(nullptr, buff.data());
  EXPECT_EQ(0, buff.size());
  EXPECT_EQ(0, buff.capacity());
  EXPECT_EQ(rmm::mr::get_current_device_resource(), buff.memory_resource());
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());
}

TYPED_TEST(DeviceBufferTest, CopyFromNullptrNonZero)
{
  // can  copy from a nullptr only if size == 0
  EXPECT_THROW(rmm::device_buffer buff(nullptr, 1, rmm::cuda_stream_view{}), rmm::logic_error);
}

TYPED_TEST(DeviceBufferTest, CopyConstructor)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_view{}, &this->mr);

  // Initialize buffer
  thrust::sequence(rmm::exec_policy(rmm::cuda_stream_default),
                   static_cast<char*>(buff.data()),
                   static_cast<char*>(buff.data()) + buff.size(),
                   0);

  rmm::device_buffer buff_copy(buff, rmm::cuda_stream_default);  // uses default MR
  EXPECT_NE(nullptr, buff_copy.data());
  EXPECT_NE(buff.data(), buff_copy.data());
  EXPECT_EQ(buff.size(), buff_copy.size());
  EXPECT_EQ(buff.capacity(), buff_copy.capacity());
  EXPECT_EQ(buff_copy.memory_resource(), rmm::mr::get_current_device_resource());
  EXPECT_TRUE(buff_copy.memory_resource()->is_equal(*rmm::mr::get_current_device_resource()));
  EXPECT_EQ(buff_copy.stream(), rmm::cuda_stream_view{});

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(rmm::cuda_stream_default),
                            static_cast<char*>(buff.data()),
                            static_cast<char*>(buff.data()) + buff.size(),
                            static_cast<char*>(buff_copy.data())));

  // now use buff's stream and MR
  rmm::device_buffer buff_copy2(buff, buff.stream(), buff.memory_resource());
  EXPECT_EQ(buff_copy2.memory_resource(), buff.memory_resource());
  EXPECT_TRUE(buff_copy2.memory_resource()->is_equal(*buff.memory_resource()));
  EXPECT_EQ(buff_copy2.stream(), buff.stream());

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(rmm::cuda_stream_default),
                            static_cast<signed char*>(buff.data()),
                            static_cast<signed char*>(buff.data()) + buff.size(),
                            static_cast<signed char*>(buff_copy.data())));
}

TYPED_TEST(DeviceBufferTest, CopyCapacityLargerThanSize)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);

  // Resizing smaller to make `size()` < `capacity()`
  auto new_size = this->size - 1;
  buff.resize(new_size, rmm::cuda_stream_default);

  thrust::sequence(rmm::exec_policy(rmm::cuda_stream_default),
                   static_cast<signed char*>(buff.data()),
                   static_cast<signed char*>(buff.data()) + buff.size(),
                   0);
  rmm::device_buffer buff_copy(buff, rmm::cuda_stream_default);
  EXPECT_NE(nullptr, buff_copy.data());
  EXPECT_NE(buff.data(), buff_copy.data());
  EXPECT_EQ(buff.size(), buff_copy.size());

  // The capacity of the copy should be equal to the `size()` of the original
  EXPECT_EQ(new_size, buff_copy.capacity());
  EXPECT_EQ(buff_copy.memory_resource(), rmm::mr::get_current_device_resource());
  EXPECT_TRUE(buff_copy.memory_resource()->is_equal(*rmm::mr::get_current_device_resource()));
  EXPECT_EQ(buff_copy.stream(), rmm::cuda_stream_view{});

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(rmm::cuda_stream_default),
                            static_cast<signed char*>(buff.data()),
                            static_cast<signed char*>(buff.data()) + buff.size(),
                            static_cast<signed char*>(buff_copy.data())));
}

TYPED_TEST(DeviceBufferTest, CopyConstructorExplicitMr)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);

  thrust::sequence(rmm::exec_policy(rmm::cuda_stream_default),
                   static_cast<signed char*>(buff.data()),
                   static_cast<signed char*>(buff.data()) + buff.size(),
                   0);
  rmm::device_buffer buff_copy(buff, this->stream, &this->mr);
  EXPECT_NE(nullptr, buff_copy.data());
  EXPECT_NE(buff.data(), buff_copy.data());
  EXPECT_EQ(buff.size(), buff_copy.size());
  EXPECT_EQ(buff.capacity(), buff_copy.capacity());
  EXPECT_EQ(buff.memory_resource(), buff_copy.memory_resource());
  EXPECT_TRUE(buff.memory_resource()->is_equal(*buff_copy.memory_resource()));
  EXPECT_NE(buff.stream(), buff_copy.stream());

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(buff_copy.stream()),
                            static_cast<signed char*>(buff.data()),
                            static_cast<signed char*>(buff.data()) + buff.size(),
                            static_cast<signed char*>(buff_copy.data())));
}

TYPED_TEST(DeviceBufferTest, CopyCapacityLargerThanSizeExplicitMr)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);

  // Resizing smaller to make `size()` < `capacity()`
  auto new_size = this->size - 1;
  buff.resize(new_size, rmm::cuda_stream_default);

  thrust::sequence(rmm::exec_policy(rmm::cuda_stream_default),
                   static_cast<signed char*>(buff.data()),
                   static_cast<signed char*>(buff.data()) + buff.size(),
                   0);
  rmm::device_buffer buff_copy(buff, this->stream, &this->mr);
  EXPECT_NE(nullptr, buff_copy.data());
  EXPECT_NE(buff.data(), buff_copy.data());
  EXPECT_EQ(buff.size(), buff_copy.size());

  // The capacity of the copy should be equal to the `size()` of the original
  EXPECT_EQ(new_size, buff_copy.capacity());
  EXPECT_NE(buff.capacity(), buff_copy.capacity());
  EXPECT_EQ(buff.memory_resource(), buff_copy.memory_resource());
  EXPECT_TRUE(buff.memory_resource()->is_equal(*buff_copy.memory_resource()));
  EXPECT_NE(buff.stream(), buff_copy.stream());

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(buff_copy.stream()),
                            static_cast<signed char*>(buff.data()),
                            static_cast<signed char*>(buff.data()) + buff.size(),
                            static_cast<signed char*>(buff_copy.data())));
}

TYPED_TEST(DeviceBufferTest, MoveConstructor)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);
  auto p        = buff.data();
  auto size     = buff.size();
  auto capacity = buff.capacity();
  auto mr       = buff.memory_resource();
  auto stream   = buff.stream();

  // New buffer should have the same contents as the original
  rmm::device_buffer buff_new(std::move(buff));
  EXPECT_NE(nullptr, buff_new.data());
  EXPECT_EQ(p, buff_new.data());
  EXPECT_EQ(size, buff_new.size());
  EXPECT_EQ(capacity, buff_new.capacity());
  EXPECT_EQ(stream, buff_new.stream());
  EXPECT_EQ(mr, buff_new.memory_resource());

  // Original buffer should be empty
  EXPECT_EQ(nullptr, buff.data());
  EXPECT_EQ(0, buff.size());
  EXPECT_EQ(0, buff.capacity());
  EXPECT_EQ(rmm::cuda_stream_default, buff.stream());
  EXPECT_NE(nullptr, buff.memory_resource());
}

TYPED_TEST(DeviceBufferTest, MoveConstructorStream)
{
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  this->stream.synchronize();
  auto p        = buff.data();
  auto size     = buff.size();
  auto capacity = buff.capacity();
  auto mr       = buff.memory_resource();
  auto stream   = buff.stream();

  // New buffer should have the same contents as the original
  rmm::device_buffer buff_new(std::move(buff));
  this->stream.synchronize();
  EXPECT_NE(nullptr, buff_new.data());
  EXPECT_EQ(p, buff_new.data());
  EXPECT_EQ(size, buff_new.size());
  EXPECT_EQ(capacity, buff_new.capacity());
  EXPECT_EQ(stream, buff_new.stream());
  EXPECT_EQ(mr, buff_new.memory_resource());

  // Original buffer should be empty
  EXPECT_EQ(nullptr, buff.data());
  EXPECT_EQ(0, buff.size());
  EXPECT_EQ(0, buff.capacity());
  EXPECT_EQ(rmm::cuda_stream_view{}, buff.stream());
  EXPECT_NE(nullptr, buff.memory_resource());
}

TYPED_TEST(DeviceBufferTest, MoveAssignmentToDefault)
{
  rmm::device_buffer from(this->size, rmm::cuda_stream_default, &this->mr);
  auto p        = from.data();
  auto size     = from.size();
  auto capacity = from.capacity();
  auto mr       = from.memory_resource();
  auto stream   = from.stream();

  rmm::device_buffer to;
  EXPECT_NO_THROW(to = std::move(from));

  // contents of `from` should be in `to`
  EXPECT_NE(nullptr, to.data());
  EXPECT_EQ(p, to.data());
  EXPECT_EQ(size, to.size());
  EXPECT_EQ(capacity, to.capacity());
  EXPECT_EQ(stream, to.stream());
  EXPECT_EQ(mr, to.memory_resource());

  // `from` should be empty
  EXPECT_EQ(nullptr, from.data());
  EXPECT_EQ(0, from.size());
  EXPECT_EQ(0, from.capacity());
  EXPECT_EQ(rmm::cuda_stream_default, from.stream());
  EXPECT_NE(nullptr, from.memory_resource());
}

TYPED_TEST(DeviceBufferTest, MoveAssignment)
{
  rmm::device_buffer from(this->size, rmm::cuda_stream_default, &this->mr);
  auto p        = from.data();
  auto size     = from.size();
  auto capacity = from.capacity();
  auto mr       = from.memory_resource();
  auto stream   = from.stream();

  rmm::device_buffer to(this->size - 1, rmm::cuda_stream_default, &this->mr);
  EXPECT_NO_THROW(to = std::move(from));

  // contents of `from` should be in `to`
  EXPECT_NE(nullptr, to.data());
  EXPECT_EQ(p, to.data());
  EXPECT_EQ(size, to.size());
  EXPECT_EQ(capacity, to.capacity());
  EXPECT_EQ(stream, to.stream());
  EXPECT_EQ(mr, to.memory_resource());

  // `from` should be empty
  EXPECT_EQ(nullptr, from.data());
  EXPECT_EQ(0, from.size());
  EXPECT_EQ(0, from.capacity());
  EXPECT_EQ(rmm::cuda_stream_default, from.stream());
  EXPECT_NE(nullptr, from.memory_resource());
}

TYPED_TEST(DeviceBufferTest, SelfMoveAssignment)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);
  auto p        = buff.data();
  auto size     = buff.size();
  auto capacity = buff.capacity();
  auto mr       = buff.memory_resource();
  auto stream   = buff.stream();

  buff = std::move(buff);  // self-move-assignment shouldn't modify the buffer
  EXPECT_NE(nullptr, buff.data());
  EXPECT_EQ(p, buff.data());
  EXPECT_EQ(size, buff.size());
  EXPECT_EQ(capacity, buff.capacity());
  EXPECT_EQ(stream, buff.stream());
  EXPECT_EQ(mr, buff.memory_resource());
}

TYPED_TEST(DeviceBufferTest, ResizeSmaller)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);

  thrust::sequence(rmm::exec_policy(rmm::cuda_stream_default),
                   static_cast<signed char*>(buff.data()),
                   static_cast<signed char*>(buff.data()) + buff.size(),
                   0);

  auto old_data = buff.data();
  rmm::device_buffer old_content(
    old_data, buff.size(), rmm::cuda_stream_default, &this->mr);  // for comparison

  auto new_size = this->size - 1;
  buff.resize(new_size, rmm::cuda_stream_default);
  EXPECT_EQ(new_size, buff.size());
  EXPECT_EQ(this->size, buff.capacity());  // Capacity should be unchanged
  // Resizing smaller means the existing allocation should remain unchanged
  EXPECT_EQ(old_data, buff.data());

  EXPECT_NO_THROW(buff.shrink_to_fit(rmm::cuda_stream_default));
  EXPECT_NE(nullptr, buff.data());
  // A reallocation should have occured
  EXPECT_NE(old_data, buff.data());
  EXPECT_EQ(new_size, buff.size());
  EXPECT_EQ(buff.capacity(), buff.size());

  EXPECT_TRUE(thrust::equal(rmm::exec_policy(rmm::cuda_stream_default),
                            static_cast<signed char*>(buff.data()),
                            static_cast<signed char*>(buff.data()) + buff.size(),
                            static_cast<signed char*>(old_content.data())));
}

TYPED_TEST(DeviceBufferTest, ResizeBigger)
{
  rmm::device_buffer buff(this->size, rmm::cuda_stream_default, &this->mr);
  auto old_data = buff.data();
  auto new_size = this->size + 1;
  buff.resize(new_size, rmm::cuda_stream_default);
  EXPECT_EQ(new_size, buff.size());
  EXPECT_EQ(new_size, buff.capacity());
  // Resizing bigger means the data should point to a new allocation
  EXPECT_NE(old_data, buff.data());
}
