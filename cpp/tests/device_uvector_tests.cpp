
/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>
#include <gtest/internal/gtest-type-util.h>

#include <cstdint>
#include <iterator>
#include <utility>

// explicit instantiation for test coverage purposes.
template class rmm::device_uvector<int32_t>;

template <typename T>
struct TypedUVectorTest : ::testing::Test {
  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept { return rmm::cuda_stream_view{}; }
};

using TestTypes = ::testing::Types<int8_t, int32_t, uint64_t, float, double>;

TYPED_TEST_SUITE(TypedUVectorTest, TestTypes);

TYPED_TEST(TypedUVectorTest, MemoryResource)
{
  rmm::device_uvector<TypeParam> vec(128, this->stream());
  EXPECT_EQ(vec.memory_resource(),
            rmm::device_async_resource_ref{rmm::mr::get_current_device_resource_ref()});
}

TYPED_TEST(TypedUVectorTest, ZeroSizeConstructor)
{
  rmm::device_uvector<TypeParam> vec(0, this->stream());
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.end(), vec.begin());
  EXPECT_TRUE(vec.is_empty());
}

TYPED_TEST(TypedUVectorTest, NonZeroSizeConstructor)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  EXPECT_EQ(vec.size(), size);
  EXPECT_EQ(vec.ssize(), size);
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_EQ(vec.end(), vec.begin() + vec.size());
  EXPECT_FALSE(vec.is_empty());
  EXPECT_NE(vec.element_ptr(0), nullptr);
}

TYPED_TEST(TypedUVectorTest, CopyConstructor)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  rmm::device_uvector<TypeParam> uv_copy(vec, this->stream());
  EXPECT_EQ(uv_copy.size(), vec.size());
  EXPECT_NE(uv_copy.data(), nullptr);
  EXPECT_EQ(uv_copy.end(), uv_copy.begin() + uv_copy.size());
  EXPECT_FALSE(uv_copy.is_empty());
  EXPECT_NE(uv_copy.element_ptr(0), nullptr);
}

TYPED_TEST(TypedUVectorTest, ResizeSmaller)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());
  auto* original_data  = vec.data();
  auto* original_begin = vec.begin();

  auto smaller_size = vec.size() - 1;
  vec.resize(smaller_size, this->stream());

  EXPECT_EQ(original_data, vec.data());
  EXPECT_EQ(original_begin, vec.begin());
  EXPECT_EQ(vec.size(), smaller_size);
  EXPECT_EQ(vec.capacity(), original_size);

  // shrink_to_fit should force a new allocation
  vec.shrink_to_fit(this->stream());
  EXPECT_EQ(vec.size(), smaller_size);
  EXPECT_EQ(vec.capacity(), smaller_size);
}

TYPED_TEST(TypedUVectorTest, ResizeLarger)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());
  auto* original_data  = vec.data();
  auto* original_begin = vec.begin();

  auto larger_size = vec.size() + 1;
  vec.resize(larger_size, this->stream());

  EXPECT_NE(vec.data(), original_data);
  EXPECT_NE(vec.begin(), original_begin);
  EXPECT_EQ(vec.size(), larger_size);
  EXPECT_EQ(vec.capacity(), larger_size);

  auto* larger_data  = vec.data();
  auto* larger_begin = vec.begin();

  // shrink_to_fit shouldn't have any effect
  vec.shrink_to_fit(this->stream());
  EXPECT_EQ(vec.size(), larger_size);
  EXPECT_EQ(vec.capacity(), larger_size);
  EXPECT_EQ(vec.data(), larger_data);
  EXPECT_EQ(vec.begin(), larger_begin);
}

TYPED_TEST(TypedUVectorTest, ReserveSmaller)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());
  auto* const original_data    = vec.data();
  auto* const original_begin   = vec.begin();
  auto const original_capacity = vec.capacity();

  auto const smaller_capacity = vec.capacity() - 1;
  vec.reserve(smaller_capacity, this->stream());

  EXPECT_EQ(vec.data(), original_data);
  EXPECT_EQ(vec.begin(), original_begin);
  EXPECT_EQ(vec.size(), original_size);
  EXPECT_EQ(vec.capacity(), original_capacity);
}

TYPED_TEST(TypedUVectorTest, ReserveLarger)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());
  vec.set_element(0, 1, this->stream());
  auto* const original_data  = vec.data();
  auto* const original_begin = vec.begin();

  auto const larger_capacity = vec.capacity() + 1;
  vec.reserve(larger_capacity, this->stream());

  EXPECT_NE(vec.data(), original_data);
  EXPECT_NE(vec.begin(), original_begin);
  EXPECT_EQ(vec.size(), original_size);
  EXPECT_EQ(vec.capacity(), larger_capacity);
  // The element should be copied
  EXPECT_EQ(vec.element(0, this->stream()), 1);
}

TYPED_TEST(TypedUVectorTest, ResizeToZero)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());
  vec.resize(0, this->stream());

  EXPECT_EQ(vec.size(), 0);
  EXPECT_TRUE(vec.is_empty());
  EXPECT_EQ(vec.capacity(), original_size);

  vec.shrink_to_fit(this->stream());
  EXPECT_EQ(vec.capacity(), 0);
}

TYPED_TEST(TypedUVectorTest, Release)
{
  auto const original_size{12345};
  rmm::device_uvector<TypeParam> vec(original_size, this->stream());

  auto* original_data = vec.data();

  rmm::device_buffer storage = vec.release();

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 0);
  EXPECT_TRUE(vec.is_empty());
  EXPECT_EQ(storage.data(), original_data);
  EXPECT_EQ(storage.size(), original_size * sizeof(TypeParam));
}

TYPED_TEST(TypedUVectorTest, ElementPointer)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    EXPECT_NE(vec.element_ptr(i), nullptr);
  }
}

TYPED_TEST(TypedUVectorTest, OOBSetElement)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  EXPECT_THROW(vec.set_element(vec.size() + 1, 42, this->stream()), rmm::out_of_range);
}

TYPED_TEST(TypedUVectorTest, OOBGetElement)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  // avoid error due to nodiscard function
  auto foo = [&]() { return vec.element(vec.size() + 1, this->stream()); };
  EXPECT_THROW(foo(), rmm::out_of_range);
}

TYPED_TEST(TypedUVectorTest, GetSetElement)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    vec.set_element(i, i, this->stream());
    EXPECT_EQ(static_cast<TypeParam>(i), vec.element(i, this->stream()));
  }
}

TYPED_TEST(TypedUVectorTest, GetSetElementAsync)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    auto init = static_cast<TypeParam>(i);
    vec.set_element_async(i, init, this->stream());
    EXPECT_EQ(init, vec.element(i, this->stream()));
  }
}

TYPED_TEST(TypedUVectorTest, SetElementZeroAsync)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    vec.set_element_to_zero_async(i, this->stream());
    EXPECT_EQ(TypeParam{0}, vec.element(i, this->stream()));
  }
}

TYPED_TEST(TypedUVectorTest, FrontBackElement)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());

  auto const first = TypeParam{42};
  auto const last  = TypeParam{13};
  vec.set_element(0, first, this->stream());
  vec.set_element(vec.size() - 1, last, this->stream());

  EXPECT_EQ(first, vec.front_element(this->stream()));
  EXPECT_EQ(last, vec.back_element(this->stream()));
}

TYPED_TEST(TypedUVectorTest, SetGetStream)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());

  EXPECT_EQ(vec.stream(), this->stream());

  rmm::cuda_stream_view const otherstream{cudaStreamPerThread};
  vec.set_stream(otherstream);

  EXPECT_EQ(vec.stream(), otherstream);
}

TYPED_TEST(TypedUVectorTest, Iterators)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());

  EXPECT_EQ(vec.begin(), vec.data());
  EXPECT_EQ(vec.cbegin(), vec.data());

  auto const* const_begin = std::as_const(vec).begin();
  EXPECT_EQ(const_begin, vec.cbegin());

  EXPECT_EQ(std::distance(vec.begin(), vec.end()), vec.size());
  EXPECT_EQ(std::distance(vec.cbegin(), vec.cend()), vec.size());

  auto const* const_end = std::as_const(vec).end();
  EXPECT_EQ(const_end, vec.cend());
}

TYPED_TEST(TypedUVectorTest, ReverseIterators)
{
  auto const size{12345};
  rmm::device_uvector<TypeParam> vec(size, this->stream());

  EXPECT_EQ(vec.rbegin().base(), vec.end());
  EXPECT_EQ(vec.crbegin().base(), vec.cend());

  EXPECT_EQ(std::distance(vec.rbegin(), vec.rend()), vec.size());
  EXPECT_EQ(std::distance(vec.crbegin(), vec.crend()), vec.size());

  EXPECT_EQ(std::distance(vec.rend(), vec.rbegin()), -static_cast<std::ptrdiff_t>(vec.size()));
  EXPECT_EQ(std::distance(vec.crend(), vec.crbegin()), -static_cast<std::ptrdiff_t>(vec.size()));

  EXPECT_EQ((vec.rbegin() + 1).base(), vec.end() - 1);
  EXPECT_EQ((vec.rend() - 1).base(), vec.begin() + 1);
}
