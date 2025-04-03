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

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>

namespace {

struct allocator_test : public ::testing::Test {
  rmm::cuda_stream stream;
};

TEST_F(allocator_test, default_resource)
{
  rmm::mr::polymorphic_allocator<int> allocator{};
  EXPECT_EQ(allocator.get_upstream_resource(),
            rmm::device_async_resource_ref{rmm::mr::get_current_device_resource()});
}

TEST_F(allocator_test, custom_resource)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::mr::polymorphic_allocator<int> allocator{&mr};
  EXPECT_EQ(allocator.get_upstream_resource(), rmm::device_async_resource_ref{mr});
}

void test_conversion(rmm::mr::polymorphic_allocator<int> /*unused*/) {}

TEST_F(allocator_test, implicit_conversion)
{
  rmm::mr::cuda_memory_resource mr;
  test_conversion(rmm::device_async_resource_ref{mr});
}

TEST_F(allocator_test, self_equality)
{
  rmm::mr::polymorphic_allocator<int> allocator{};
  EXPECT_EQ(allocator, allocator);
  EXPECT_FALSE(allocator != allocator);
}

TEST_F(allocator_test, equal_resources)
{
  rmm::mr::cuda_memory_resource mr0;
  rmm::mr::polymorphic_allocator<int> alloc0{&mr0};

  rmm::mr::cuda_memory_resource mr1;
  rmm::mr::polymorphic_allocator<int> alloc1{&mr1};
  EXPECT_EQ(alloc0, alloc1);
  EXPECT_FALSE(alloc0 != alloc1);
}

TEST_F(allocator_test, unequal_resources)
{
  rmm::mr::managed_memory_resource mr0;
  rmm::mr::polymorphic_allocator<int> alloc0{&mr0};

  rmm::mr::cuda_memory_resource mr1;
  rmm::mr::polymorphic_allocator<int> alloc1{&mr1};
  EXPECT_NE(alloc0, alloc1);
}

TEST_F(allocator_test, copy_ctor_same_type)
{
  rmm::mr::polymorphic_allocator<int> alloc0;
  rmm::mr::polymorphic_allocator<int> alloc1{alloc0};
  EXPECT_EQ(alloc0, alloc1);
  EXPECT_EQ(alloc0.get_upstream_resource(), alloc1.get_upstream_resource());
}

TEST_F(allocator_test, copy_ctor_different_type)
{
  rmm::mr::polymorphic_allocator<int> alloc0;
  rmm::mr::polymorphic_allocator<double> alloc1{alloc0};
  EXPECT_EQ(alloc0, alloc1);
  EXPECT_EQ(alloc0.get_upstream_resource(), alloc1.get_upstream_resource());
}

TEST_F(allocator_test, rebind)
{
  using Allocator = rmm::mr::polymorphic_allocator<int>;
  Allocator alloc0;

  using Rebound = std::allocator_traits<Allocator>::rebind_alloc<double>;

  EXPECT_TRUE((std::is_same_v<std::allocator_traits<Rebound>::value_type, double>));
}

TEST_F(allocator_test, allocate_deallocate)
{
  rmm::mr::polymorphic_allocator<int> allocator{};
  const auto size{1000};
  auto* ptr = allocator.allocate(size, stream);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(allocator.deallocate(ptr, size, stream));
}

}  // namespace
