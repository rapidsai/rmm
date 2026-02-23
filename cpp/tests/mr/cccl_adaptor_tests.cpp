/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/is_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using cuda_mr = rmm::mr::cuda_memory_resource;
using rmm::mr::logging_resource_adaptor;
using rmm::mr::pool_memory_resource;

// static property checks
static_assert(cuda::mr::resource_with<logging_resource_adaptor, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<pool_memory_resource, cuda::mr::device_accessible>);

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
    if constexpr (std::is_same_v<AdaptorType, logging_resource_adaptor>) {
      return AdaptorType{cuda, "rmm_cccl_adaptor_test.txt"};
    } else if constexpr (std::is_same_v<AdaptorType, pool_memory_resource>) {
      return AdaptorType{cuda, 0};
    }
  }
};

using cccl_adaptors = ::testing::Types<logging_resource_adaptor, pool_memory_resource>;

TYPED_TEST_SUITE(CcclAdaptorTest, cccl_adaptors);

TYPED_TEST(CcclAdaptorTest, Equality)
{
  auto copy = this->mr;
  EXPECT_TRUE(this->mr == copy);

  auto other = this->make_mr();
  EXPECT_FALSE(this->mr == other);
}

TYPED_TEST(CcclAdaptorTest, GetUpstreamResource)
{
  rmm::device_async_resource_ref expected{this->cuda};
  EXPECT_TRUE(this->mr.get_upstream_resource() == expected);
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

// ---------------------------------------------------------------------------
// CcclMrRef typed-test instantiations
// ---------------------------------------------------------------------------

INSTANTIATE_TYPED_TEST_SUITE_P(LoggingAdaptor,
                               CcclMrRefTest,
                               CcclAdaptorTest<logging_resource_adaptor>);
INSTANTIATE_TYPED_TEST_SUITE_P(LoggingAdaptor,
                               CcclMrRefAllocationTest,
                               CcclAdaptorTest<logging_resource_adaptor>);

}  // namespace rmm::test
