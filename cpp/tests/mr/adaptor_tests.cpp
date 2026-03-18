/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/error.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/is_resource_adaptor.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <type_traits>

using cuda_mr = rmm::mr::cuda_memory_resource;
using rmm::mr::aligned_resource_adaptor;
using rmm::mr::failure_callback_resource_adaptor;
using rmm::mr::limiting_resource_adaptor;
using rmm::mr::thread_safe_resource_adaptor;

// explicit instantiations for test coverage purposes
template class rmm::mr::failure_callback_resource_adaptor<>;

namespace rmm::test {

using adaptors = ::testing::Types<failure_callback_resource_adaptor<>,
                                  limiting_resource_adaptor,
                                  thread_safe_resource_adaptor>;

// static property checks
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::aligned_resource_adaptor,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::failure_callback_resource_adaptor<>,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::limiting_resource_adaptor,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::logging_resource_adaptor,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::statistics_resource_adaptor,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::thread_safe_resource_adaptor,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<rmm::mr::tracking_resource_adaptor,
                                                  cuda::mr::device_accessible>);

template <typename MemoryResourceType>
struct AdaptorTest : public ::testing::Test {
  using adaptor_type = MemoryResourceType;
  cuda_mr cuda{};
  std::shared_ptr<adaptor_type> mr;

  AdaptorTest() : mr{make_adaptor(&cuda)} {}

  auto make_adaptor(cuda_mr* upstream)
  {
    if constexpr (std::is_same_v<adaptor_type, failure_callback_resource_adaptor<>>) {
      return std::make_shared<adaptor_type>(
        *upstream,
        []([[maybe_unused]] std::size_t bytes, [[maybe_unused]] void* arg) { return false; },
        nullptr);
    } else if constexpr (std::is_same_v<adaptor_type, limiting_resource_adaptor>) {
      return std::make_shared<adaptor_type>(*upstream, 64_MiB);
    } else if constexpr (std::is_same_v<adaptor_type, thread_safe_resource_adaptor>) {
      return std::make_shared<adaptor_type>(*upstream);
    } else {
      return std::make_shared<adaptor_type>(*upstream);
    }
  }
};

TYPED_TEST_SUITE(AdaptorTest, adaptors);

TYPED_TEST(AdaptorTest, Equality)
{
  EXPECT_EQ(*this->mr, *this->mr);

  {
    auto other_mr = this->make_adaptor(&this->cuda);
    if constexpr (std::is_same_v<TypeParam, failure_callback_resource_adaptor<>> or
                  std::is_same_v<TypeParam, limiting_resource_adaptor> or
                  std::is_same_v<TypeParam, thread_safe_resource_adaptor>) {
      // shared_resource equality: two distinct constructions are NOT equal
      EXPECT_NE(*this->mr, *other_mr);
    } else {
      EXPECT_EQ(*this->mr, *other_mr);
    }
  }
}

TYPED_TEST(AdaptorTest, GetUpstreamResource)
{
  rmm::device_async_resource_ref expected{this->cuda};
  EXPECT_EQ(this->mr->get_upstream_resource(), expected);
  EXPECT_TRUE(rmm::mr::is_resource_adaptor<decltype(*this->mr)>);
}

TYPED_TEST(AdaptorTest, AllocFree)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->mr->allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(this->mr->deallocate_sync(ptr, 1024));
}

}  // namespace rmm::test
