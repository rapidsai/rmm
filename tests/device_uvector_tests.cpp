
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

#include <gtest/gtest.h>
#include <gtest/internal/gtest-type-util.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

template <typename T>
struct TypedUVectorTest : ::testing::Test {
  rmm::cuda_stream_view stream() const noexcept { return rmm::cuda_stream_view{}; }
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
  EXPECT_NE(uv.element_ptr(0), nullptr);
}

TYPED_TEST(TypedUVectorTest, CopyConstructor)
{
  rmm::device_uvector<TypeParam> uv(12345, this->stream());
  rmm::device_uvector<TypeParam> uv_copy(uv, this->stream());
  EXPECT_EQ(uv_copy.size(), uv.size());
  EXPECT_NE(uv_copy.data(), nullptr);
  EXPECT_EQ(uv_copy.end(), uv_copy.begin() + uv_copy.size());
  EXPECT_FALSE(uv_copy.is_empty());
  EXPECT_NE(uv_copy.element_ptr(0), nullptr);
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

TYPED_TEST(TypedUVectorTest, ElementPointer)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());
  for (std::size_t i = 0; i < uv.size(); ++i) {
    EXPECT_NE(uv.element_ptr(i), nullptr);
  }
}

TYPED_TEST(TypedUVectorTest, OOBSetElement)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());
  EXPECT_THROW(uv.set_element(uv.size() + 1, 42, this->stream()), rmm::out_of_range);
}

TYPED_TEST(TypedUVectorTest, OOBGetElement)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());
  EXPECT_THROW(uv.element(uv.size() + 1, this->stream()), rmm::out_of_range);
}

TYPED_TEST(TypedUVectorTest, GetSetElement)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());
  for (std::size_t i = 0; i < uv.size(); ++i) {
    uv.set_element(i, i, this->stream());
    EXPECT_EQ(static_cast<TypeParam>(i), uv.element(i, this->stream()));
  }
}

TYPED_TEST(TypedUVectorTest, GetSetElementAsync)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());
  for (std::size_t i = 0; i < uv.size(); ++i) {
    uv.set_element_async(i, i, this->stream());
    EXPECT_EQ(static_cast<TypeParam>(i), uv.element(i, this->stream()));
  }
}

TYPED_TEST(TypedUVectorTest, FrontBackElement)
{
  auto size = 12345;
  rmm::device_uvector<TypeParam> uv(size, this->stream());

  auto first = TypeParam{42};
  auto last  = TypeParam{13};
  uv.set_element(0, first, this->stream());
  uv.set_element(uv.size() - 1, last, this->stream());

  EXPECT_EQ(first, uv.front_element(this->stream()));
  EXPECT_EQ(last, uv.back_element(this->stream()));
}
