/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/mr/is_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using cuda_mr = rmm::mr::cuda_memory_resource;
using rmm::mr::aligned_resource_adaptor;
using rmm::mr::arena_memory_resource;
using rmm::mr::binning_memory_resource;
using rmm::mr::callback_memory_resource;
using rmm::mr::failure_callback_resource_adaptor;
using rmm::mr::fixed_size_memory_resource;
using rmm::mr::logging_resource_adaptor;
using rmm::mr::pool_memory_resource;
using rmm::mr::prefetch_resource_adaptor;
using rmm::mr::statistics_resource_adaptor;
using rmm::mr::thread_safe_resource_adaptor;
using rmm::mr::tracking_resource_adaptor;

// static property checks
static_assert(cuda::mr::resource_with<aligned_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<arena_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<binning_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<callback_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<failure_callback_resource_adaptor<>, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<fixed_size_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<logging_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<pool_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<prefetch_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<statistics_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<thread_safe_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>);

namespace rmm::test {

// ---------------------------------------------------------------------------
// Typed fixture for CCCL-style (non-template, shared_resource-based) adaptors
//
// Each adaptor type must provide:
//   - Value construction (no raw pointer upstream)
//   - operator== with shared ownership semantics
//   - get_upstream_resource()
//   - SharedOwnership: alloc through one copy, dealloc through another
// ---------------------------------------------------------------------------

template <typename AdaptorType>
struct CcclAdaptorTest : public ::testing::Test {
  cuda_mr cuda{};
  AdaptorType mr{make_mr()};
  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};

  AdaptorType make_mr()
  {
    if constexpr (std::is_same_v<AdaptorType, aligned_resource_adaptor>) {
      return AdaptorType{cuda};
    } else if constexpr (std::is_same_v<AdaptorType, binning_memory_resource>) {
      return AdaptorType{cuda, 18, 22};
    } else if constexpr (std::is_same_v<AdaptorType, failure_callback_resource_adaptor<>>) {
      return AdaptorType{cuda, [](std::size_t, void*) { return false; }, nullptr};
    } else if constexpr (std::is_same_v<AdaptorType, fixed_size_memory_resource>) {
      return AdaptorType{cuda};
    } else if constexpr (std::is_same_v<AdaptorType, logging_resource_adaptor>) {
      return AdaptorType{cuda, "rmm_cccl_adaptor_test.txt"};
    } else if constexpr (std::is_same_v<AdaptorType, pool_memory_resource>) {
      return AdaptorType{cuda, 0};
    } else if constexpr (std::is_same_v<AdaptorType, prefetch_resource_adaptor>) {
      // TODO: Wrap a managed resource here.
      return AdaptorType{cuda};
    } else if constexpr (std::is_same_v<AdaptorType, statistics_resource_adaptor>) {
      return AdaptorType{cuda};
    } else if constexpr (std::is_same_v<AdaptorType, thread_safe_resource_adaptor>) {
      return AdaptorType{cuda};
    } else if constexpr (std::is_same_v<AdaptorType, tracking_resource_adaptor>) {
      return AdaptorType{cuda};
    }
  }
};

using cccl_adaptors = ::testing::Types<aligned_resource_adaptor,
                                       binning_memory_resource,
                                       failure_callback_resource_adaptor<>,
                                       fixed_size_memory_resource,
                                       logging_resource_adaptor,
                                       pool_memory_resource,
                                       prefetch_resource_adaptor,
                                       statistics_resource_adaptor,
                                       thread_safe_resource_adaptor,
                                       tracking_resource_adaptor>;

TYPED_TEST_SUITE(CcclAdaptorTest, cccl_adaptors);

TYPED_TEST(CcclAdaptorTest, Equality)
{
  auto copy = this->mr;
  EXPECT_EQ(this->mr, copy);

  auto other = this->make_mr();
  EXPECT_NE(this->mr, other);
}

TYPED_TEST(CcclAdaptorTest, GetUpstreamResource)
{
  rmm::device_async_resource_ref expected{this->cuda};
  EXPECT_EQ(this->mr.get_upstream_resource(), expected);
  EXPECT_TRUE(rmm::mr::is_resource_adaptor<TypeParam>);
}

TYPED_TEST(CcclAdaptorTest, AllocFree)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->mr.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(this->mr.deallocate_sync(ptr, 1024));
}

TYPED_TEST(CcclAdaptorTest, SharedOwnership)
{
  auto copy = this->mr;
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->mr.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, 1024));
}

TEST(AlignedAdaptorTest, ThrowOnInvalidAllocationAlignment)
{
  cuda_mr cuda{};
  EXPECT_THROW((aligned_resource_adaptor{cuda, 255}), rmm::logic_error);
  EXPECT_NO_THROW((aligned_resource_adaptor{cuda, 256}));
  EXPECT_THROW((aligned_resource_adaptor{cuda, 768}), rmm::logic_error);
}

// arena_memory_resource: does not expose get_upstream_resource(); test separately.
TEST(ArenaMRAdaptorTest, EqualityAndSharedOwnership)
{
  cuda_mr cuda{};
  arena_memory_resource mr{cuda};
  auto copy = mr;
  EXPECT_EQ(mr, copy);

  arena_memory_resource other{cuda};
  EXPECT_NE(mr, other);

  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, 1024));
}

// callback_memory_resource: no upstream resource; test separately.
TEST(CallbackMRAdaptorTest, EqualityAndSharedOwnership)
{
  cuda_mr cuda{};
  rmm::device_async_resource_ref upstream{cuda};

  auto alloc_cb = [](std::size_t bytes, rmm::cuda_stream_view stream, void* arg) {
    return static_cast<rmm::device_async_resource_ref*>(arg)->allocate(stream, bytes);
  };
  auto dealloc_cb = [](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void* arg) {
    static_cast<rmm::device_async_resource_ref*>(arg)->deallocate(stream, ptr, bytes);
  };

  callback_memory_resource mr{alloc_cb, dealloc_cb, &upstream, &upstream};
  auto copy = mr;
  EXPECT_EQ(mr, copy);

  callback_memory_resource other{alloc_cb, dealloc_cb, &upstream, &upstream};
  EXPECT_NE(mr, other);

  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, 1024));
}

}  // namespace rmm::test
