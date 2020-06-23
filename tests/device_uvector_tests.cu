/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "gtest/internal/gtest-type-util.h"

#include <rmm/device_uvector.hpp>

#include <thrust/logical.h>

template <typename T>
struct TypedUVectorTest : ::testing::Test {
  cudaStream_t stream() const noexcept { return cudaStream_t{0}; }
};

using TestTypes = ::testing::Types<int8_t, int32_t, uint64_t, float, double>;

TYPED_TEST_CASE(TypedUVectorTest, TestTypes);

TYPED_TEST(TypedUVectorTest, ZeroSizeConstructor)
{
  rmm::device_uvector<TypeParam> uv(0, this->stream());
  EXPECT_EQ(uv.size(), 0);
  EXPECT_EQ(uv.end(), uv.begin());
  EXPECT_TRUE(uv.is_empty());
}

TYPED_TEST(TypedUVectorTest, NonZeroSizeConstructor)
{
  rmm::device_uvector<TypeParam> uv(12345, this->stream());
  EXPECT_EQ(uv.size(), 12345);
  EXPECT_NE(uv.data(), nullptr);
  EXPECT_EQ(uv.end(), uv.begin() + uv.size());
  EXPECT_FALSE(uv.is_empty());
}

TYPED_TEST(TypedUVectorTest, ResizeSmaller)
{
  auto original_size = 12345;
  rmm::device_uvector<TypeParam> uv(original_size, this->stream());
  auto original_data  = uv.data();
  auto original_begin = uv.begin();

  auto smaller_size = uv.size() - 1;
  uv.resize(smaller_size, this->stream());

  EXPECT_EQ(original_data, uv.data());
  EXPECT_EQ(original_begin, uv.begin());
  EXPECT_EQ(uv.size(), smaller_size);
  EXPECT_EQ(uv.capacity(), original_size);

  // shrink_to_fit should force a new allocation
  uv.shrink_to_fit(this->stream());
  EXPECT_EQ(uv.size(), smaller_size);
  EXPECT_EQ(uv.capacity(), smaller_size);
}

TYPED_TEST(TypedUVectorTest, ResizeLarger)
{
  auto original_size = 12345;
  rmm::device_uvector<TypeParam> uv(original_size, this->stream());
  auto original_data  = uv.data();
  auto original_begin = uv.begin();

  auto larger_size = uv.size() + 1;
  uv.resize(larger_size, this->stream());

  EXPECT_NE(uv.data(), original_data);
  EXPECT_NE(uv.begin(), original_begin);
  EXPECT_EQ(uv.size(), larger_size);
  EXPECT_EQ(uv.capacity(), larger_size);

  auto larger_data  = uv.data();
  auto larger_begin = uv.begin();

  // shrink_to_fit shouldn't have any effect
  uv.shrink_to_fit(this->stream());
  EXPECT_EQ(uv.size(), larger_size);
  EXPECT_EQ(uv.capacity(), larger_size);
  EXPECT_EQ(uv.data(), larger_data);
  EXPECT_EQ(uv.begin(), larger_begin);
}

TYPED_TEST(TypedUVectorTest, ResizeToZero)
{
  auto original_size = 12345;
  rmm::device_uvector<TypeParam> uv(original_size, this->stream());
  uv.resize(0, this->stream());

  EXPECT_EQ(uv.size(), 0);
  EXPECT_TRUE(uv.is_empty());
  EXPECT_EQ(uv.capacity(), original_size);

  uv.shrink_to_fit(this->stream());
  EXPECT_EQ(uv.capacity(), 0);
}

TYPED_TEST(TypedUVectorTest, Release)
{
  auto original_size = 12345;
  rmm::device_uvector<TypeParam> uv(original_size, this->stream());

  auto original_data = uv.data();

  rmm::device_buffer storage = uv.release();

  EXPECT_EQ(uv.size(), 0);
  EXPECT_EQ(uv.capacity(), 0);
  EXPECT_TRUE(uv.is_empty());
  EXPECT_EQ(storage.data(), original_data);
  EXPECT_EQ(storage.size(), original_size * sizeof(TypeParam));
}
