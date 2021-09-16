/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

namespace {

struct allocator_test : public ::testing::Test {
  rmm::cuda_stream stream{};
  rmm::mr::polymorphic_allocator<int> allocator{};
};

TEST_F(allocator_test, factory)
{
  using Adaptor = rmm::mr::stream_allocator_adaptor<decltype(allocator)>;
  auto adapted  = rmm::mr::make_stream_allocator_adaptor(allocator, stream);
  static_assert((std::is_same<decltype(adapted), Adaptor>::value));
  EXPECT_EQ(adapted.underlying_allocator(), allocator);
  EXPECT_EQ(adapted.stream(), stream);
}

TEST_F(allocator_test, self_equality)
{
  auto adapted = rmm::mr::make_stream_allocator_adaptor(allocator, stream);
  EXPECT_EQ(adapted, adapted);
  EXPECT_FALSE(adapted != adapted);
}

TEST_F(allocator_test, equal_allocators)
{
  rmm::mr::polymorphic_allocator<int> alloc0;
  auto adapted0 = rmm::mr::make_stream_allocator_adaptor(alloc0, stream);

  rmm::mr::polymorphic_allocator<int> alloc1;
  auto adapted1 = rmm::mr::make_stream_allocator_adaptor(alloc1, stream);

  EXPECT_EQ(adapted0, adapted1);
  EXPECT_FALSE(adapted0 != adapted1);
}

TEST_F(allocator_test, unequal_resources)
{
  rmm::mr::cuda_memory_resource mr0;
  rmm::mr::polymorphic_allocator<int> alloc0{&mr0};
  auto adapted0 = rmm::mr::make_stream_allocator_adaptor(alloc0, stream);

  rmm::mr::managed_memory_resource mr1;
  rmm::mr::polymorphic_allocator<int> alloc1{&mr1};
  auto adapted1 = rmm::mr::make_stream_allocator_adaptor(alloc1, stream);

  EXPECT_NE(adapted0, adapted1);
}

TEST_F(allocator_test, copy_ctor_same_type)
{
  rmm::mr::polymorphic_allocator<int> alloc0;
  auto adapted0 = rmm::mr::make_stream_allocator_adaptor(alloc0, stream);

  using Adaptor = rmm::mr::stream_allocator_adaptor<decltype(alloc0)>;
  Adaptor adapted1{adapted0};

  EXPECT_EQ(adapted0, adapted1);
}

TEST_F(allocator_test, copy_ctor_different_type)
{
  rmm::mr::polymorphic_allocator<int> alloc0;
  auto adapted0 = rmm::mr::make_stream_allocator_adaptor(alloc0, stream);

  using Adaptor = rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<double>>;
  Adaptor adapted1{adapted0};

  EXPECT_EQ(adapted0, adapted1);
}

TEST_F(allocator_test, rebind)
{
  auto adapted  = rmm::mr::make_stream_allocator_adaptor(allocator, stream);
  using Rebound = std::allocator_traits<decltype(adapted)>::rebind_alloc<double>;
  static_assert((std::is_same<std::allocator_traits<Rebound>::value_type, double>::value));
  static_assert(
    std::is_same<Rebound,
                 rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<double>>>::value);

  Rebound rebound{adapted};
}

TEST_F(allocator_test, allocate_deallocate)
{
  auto adapted = rmm::mr::make_stream_allocator_adaptor(allocator, stream);
  auto const size{1000};
  auto* ptr = adapted.allocate(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(adapted.deallocate(ptr, size));
}

}  // namespace
