/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../mock_resource.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/cccl_adaptors.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/detail/device_memory_resource_view.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace rmm::test {
namespace {

/**
 * @brief Test that device_memory_resource_view can be constructed from a pointer
 */
TEST(DeviceMemoryResourceViewTest, ConstructFromPointer)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::mr::detail::device_memory_resource_view view{&mr};

  // Should be able to get the pointer back
  EXPECT_EQ(view.get(), &mr);
}

/**
 * @brief Test that device_memory_resource_view rejects null pointers
 */
TEST(DeviceMemoryResourceViewTest, RejectsNullPointer)
{
  EXPECT_THROW(rmm::mr::detail::device_memory_resource_view view{nullptr}, rmm::logic_error);
}

/**
 * @brief Test that device_memory_resource_view is copyable
 */
TEST(DeviceMemoryResourceViewTest, IsCopyable)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::mr::detail::device_memory_resource_view view1{&mr};

  // Copy construction
  auto view2 = view1;
  EXPECT_EQ(view2.get(), &mr);

  // Copy assignment
  rmm::mr::cuda_memory_resource mr2;
  rmm::mr::detail::device_memory_resource_view view3{&mr2};
  view3 = view1;
  EXPECT_EQ(view3.get(), &mr);
}

/**
 * @brief Test that device_memory_resource_view is movable
 */
TEST(DeviceMemoryResourceViewTest, IsMovable)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::mr::detail::device_memory_resource_view view1{&mr};

  // Move construction
  auto view2 = std::move(view1);
  EXPECT_EQ(view2.get(), &mr);

  // Move assignment
  rmm::mr::cuda_memory_resource mr2;
  rmm::mr::detail::device_memory_resource_view view3{&mr2};
  rmm::mr::detail::device_memory_resource_view view4{&mr};
  view3 = std::move(view4);
  EXPECT_EQ(view3.get(), &mr);
}

/**
 * @brief Test that device_memory_resource_view properly compares equal views
 */
TEST(DeviceMemoryResourceViewTest, EqualityComparison)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::mr::detail::device_memory_resource_view view1{&mr};
  rmm::mr::detail::device_memory_resource_view view2{&mr};

  // Views wrapping the same resource should be equal
  EXPECT_EQ(view1, view2);
  EXPECT_FALSE(view1 != view2);
}

/**
 * @brief Test that device_memory_resource_view properly compares unequal views
 */
TEST(DeviceMemoryResourceViewTest, InequalityComparison)
{
  rmm::mr::cuda_memory_resource mr1;
  rmm::mr::cuda_memory_resource mr2;
  rmm::mr::cuda_async_memory_resource mr3;
  rmm::mr::detail::device_memory_resource_view view1{&mr1};
  rmm::mr::detail::device_memory_resource_view view2{&mr2};
  rmm::mr::detail::device_memory_resource_view view3{&mr3};

  // Views wrapping different resources should be equal iff the resources compare equal
  EXPECT_EQ(mr1, mr2);
  EXPECT_EQ(view1, view2);
  EXPECT_TRUE(view1 == view2);
  EXPECT_NE(mr1, mr3);
  EXPECT_NE(view1, view3);
  EXPECT_FALSE(view1 == view3);
}

/**
 * @brief Test that device_memory_resource_view forwards allocate_sync calls
 */
TEST(DeviceMemoryResourceViewTest, ForwardsAllocateSync)
{
  mock_resource mr;
  rmm::mr::detail::device_memory_resource_view view{&mr};

  void* expected_ptr = reinterpret_cast<void*>(0x1234);
  EXPECT_CALL(mr, do_allocate(100, cuda_stream_view{}))
    .Times(1)
    .WillOnce(::testing::Return(expected_ptr));

  void* ptr = view.allocate_sync(100);
  EXPECT_EQ(ptr, expected_ptr);
}

/**
 * @brief Test that device_memory_resource_view forwards deallocate_sync calls
 */
TEST(DeviceMemoryResourceViewTest, ForwardsDeallocateSync)
{
  mock_resource mr;
  rmm::mr::detail::device_memory_resource_view view{&mr};

  void* ptr = reinterpret_cast<void*>(0x1234);
  EXPECT_CALL(mr, do_deallocate(ptr, 100, cuda_stream_view{})).Times(1);

  view.deallocate_sync(ptr, 100);
}

/**
 * @brief Test that device_memory_resource_view forwards async allocate calls
 */
TEST(DeviceMemoryResourceViewTest, ForwardsAllocateAsync)
{
  mock_resource mr;
  rmm::mr::detail::device_memory_resource_view view{&mr};

  cuda_stream_view stream{};
  void* expected_ptr = reinterpret_cast<void*>(0x5678);
  EXPECT_CALL(mr, do_allocate(200, stream)).Times(1).WillOnce(::testing::Return(expected_ptr));

  void* ptr = view.allocate(stream, 200);
  EXPECT_EQ(ptr, expected_ptr);
}

/**
 * @brief Test that device_memory_resource_view forwards async deallocate calls
 */
TEST(DeviceMemoryResourceViewTest, ForwardsDeallocateAsync)
{
  mock_resource mr;
  rmm::mr::detail::device_memory_resource_view view{&mr};

  cuda_stream_view stream{};
  void* ptr = reinterpret_cast<void*>(0x5678);
  EXPECT_CALL(mr, do_deallocate(ptr, 200, stream)).Times(1);

  view.deallocate(stream, ptr, 200);
}

/**
 * @brief Test that cccl_resource_ref can be constructed from a raw pointer
 */
TEST(DeviceMemoryResourceViewTest, CcclResourceRefFromPointer)
{
  rmm::mr::cuda_memory_resource mr;

  // This should compile and work thanks to the new constructor
  rmm::device_resource_ref ref{&mr};

  // The ref should be usable for allocations
  void* ptr = ref.allocate_sync(100);
  EXPECT_NE(ptr, nullptr);
  ref.deallocate_sync(ptr, 100);
}

/**
 * @brief Test that cccl_async_resource_ref can be constructed from a raw pointer
 */
TEST(DeviceMemoryResourceViewTest, CcclAsyncResourceRefFromPointer)
{
  rmm::mr::cuda_memory_resource mr;

  // This should compile and work thanks to the new constructor
  rmm::device_async_resource_ref ref{&mr};

  // The ref should be usable for allocations
  void* ptr = ref.allocate_sync(100);
  EXPECT_NE(ptr, nullptr);
  ref.deallocate_sync(ptr, 100);

  // And for async allocations
  cuda_stream_view stream{};
  ptr = ref.allocate(stream, 200);
  EXPECT_NE(ptr, nullptr);
  ref.deallocate(stream, ptr, 200);
}

/**
 * @brief Test that resource_refs constructed from pointers are copyable
 */
TEST(DeviceMemoryResourceViewTest, ResourceRefFromPointerIsCopyable)
{
  rmm::mr::cuda_memory_resource mr;
  rmm::device_async_resource_ref ref1{&mr};

  // Copy construction
  auto ref2 = ref1;

  // Both refs should work
  void* ptr1 = ref1.allocate_sync(100);
  void* ptr2 = ref2.allocate_sync(100);

  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(ptr2, nullptr);
  EXPECT_NE(ptr1, ptr2);

  ref1.deallocate_sync(ptr1, 100);
  ref2.deallocate_sync(ptr2, 100);
}

/**
 * @brief Test that the view maintains the correct lifecycle semantics (non-owning)
 */
TEST(DeviceMemoryResourceViewTest, NonOwningSemantics)
{
  auto mr_ptr = std::make_unique<rmm::mr::cuda_memory_resource>();

  // Destroying the view should NOT destroy the resource
  {
    rmm::mr::detail::device_memory_resource_view view{mr_ptr.get()};
  }

  // The underlying resource should still be valid
  void* ptr = mr_ptr->allocate_sync(100);
  EXPECT_NE(ptr, nullptr);
  mr_ptr->deallocate_sync(ptr, 100);
}

}  // namespace
}  // namespace rmm::test
