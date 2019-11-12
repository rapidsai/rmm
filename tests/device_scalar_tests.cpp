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

#include <rmm/device_scalar.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <chrono>
#include <cstddef>
#include <random>

void sync_stream(cudaStream_t stream) {
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
}

template <typename T>
struct DeviceScalarTest : public ::testing::Test {
  cudaStream_t stream{};
  rmm::mr::device_memory_resource* mr{rmm::mr::get_default_resource()};
  T value{};
  std::default_random_engine generator{};
  std::uniform_int_distribution<T> distribution{
      std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};

  DeviceScalarTest() { value = distribution(generator); }

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override {
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  };
};

using Types = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;

TYPED_TEST_CASE(DeviceScalarTest, Types);

TYPED_TEST(DeviceScalarTest, DefaultUninitialized) {
  rmm::device_scalar<TypeParam> scalar{};
  EXPECT_NE(nullptr, scalar.data());
}

TYPED_TEST(DeviceScalarTest, InitialValue) {
  rmm::device_scalar<TypeParam> scalar{this->value, this->stream, this->mr};
  EXPECT_NE(nullptr, scalar.data());
  EXPECT_EQ(this->value, scalar.value());
}

TYPED_TEST(DeviceScalarTest, CopyCtor) {
  rmm::device_scalar<TypeParam> scalar{this->value, this->stream, this->mr};
  EXPECT_NE(nullptr, scalar.data());
  EXPECT_EQ(this->value, scalar.value());

  rmm::device_scalar<TypeParam> copy{scalar};
  EXPECT_NE(nullptr, copy.data());
  EXPECT_NE(copy.data(), scalar.data());
  EXPECT_EQ(copy.value(), scalar.value());
}

TYPED_TEST(DeviceScalarTest, MoveCtor) {
  rmm::device_scalar<TypeParam> scalar{this->value, this->stream, this->mr};
  EXPECT_NE(nullptr, scalar.data());
  EXPECT_EQ(this->value, scalar.value());

  auto original_pointer = scalar.data();
  auto original_value = scalar.value();

  rmm::device_scalar<TypeParam> moved_to{std::move(scalar)};
  EXPECT_NE(nullptr, moved_to.data());
  EXPECT_EQ(moved_to.data(), original_pointer);
  EXPECT_EQ(moved_to.value(), original_value);
  EXPECT_EQ(nullptr, scalar.data());
}

TYPED_TEST(DeviceScalarTest, SetValue) {
  rmm::device_scalar<TypeParam> scalar{this->value, this->stream, this->mr};
  EXPECT_NE(nullptr, scalar.data());

  auto expected = this->distribution(this->generator);

  scalar.set_value(expected);
  EXPECT_EQ(expected, scalar.value());
}
