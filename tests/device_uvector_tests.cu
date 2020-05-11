/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

template <typename T>
struct TypedUVectorTest : ::testing::Test {};

// An example, non-primitive, trivially copyable type
struct trivial_type {
  int v;
  float f;
};

using TestTypes = ::testing::Types<int32_t, float, trivial_type>;

TYPED_TEST_CASE(TypedUVectorTest, TestTypes);

TYPED_TEST(TypedUVectorTest, DefaultConstructed) {
  rmm::device_uvector<TypeParam> uv;
  EXPECT_EQ(uv.size(), 0);
  EXPECT_EQ(uv.data(), nullptr);
  EXPECT_EQ(uv.begin(), uv.end());
  EXPECT_TRUE(uv.is_empty());
  EXPECT_NE(uv.memory_resource(), nullptr);
}
