/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/is_resource_adaptor.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/owning_wrapper.hpp>
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
using rmm::mr::logging_resource_adaptor;
using rmm::mr::statistics_resource_adaptor;
using rmm::mr::thread_safe_resource_adaptor;
using rmm::mr::tracking_resource_adaptor;
using owning_wrapper = rmm::mr::owning_wrapper<aligned_resource_adaptor<cuda_mr>, cuda_mr>;

// explicit instantiations for test coverage purposes
template class rmm::mr::aligned_resource_adaptor<cuda_mr>;
template class rmm::mr::failure_callback_resource_adaptor<cuda_mr>;
template class rmm::mr::limiting_resource_adaptor<cuda_mr>;
template class rmm::mr::statistics_resource_adaptor<cuda_mr>;
template class rmm::mr::thread_safe_resource_adaptor<cuda_mr>;
template class rmm::mr::tracking_resource_adaptor<cuda_mr>;

namespace rmm::test {

using adaptors = ::testing::Types<aligned_resource_adaptor<cuda_mr>,
                                  failure_callback_resource_adaptor<cuda_mr>,
                                  limiting_resource_adaptor<cuda_mr>,
                                  owning_wrapper,
                                  statistics_resource_adaptor<cuda_mr>,
                                  thread_safe_resource_adaptor<cuda_mr>,
                                  tracking_resource_adaptor<cuda_mr>>;

// static property checks
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::aligned_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(
  rmm::detail::polyfill::resource_with<rmm::mr::failure_callback_resource_adaptor<cuda_mr>,
                                       cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::limiting_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::async_resource_with<rmm::mr::logging_resource_adaptor,
                                                         cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::owning_wrapper<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::statistics_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::thread_safe_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::tracking_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);

template <typename MemoryResourceType>
struct AdaptorTest : public ::testing::Test {
  using adaptor_type = MemoryResourceType;
  cuda_mr cuda{};
  std::shared_ptr<adaptor_type> mr;

  AdaptorTest() : mr{make_adaptor(&cuda)} {}

  auto make_adaptor(cuda_mr* upstream)
  {
    if constexpr (std::is_same_v<adaptor_type, failure_callback_resource_adaptor<cuda_mr>>) {
      return std::make_shared<adaptor_type>(
        upstream,
        []([[maybe_unused]] std::size_t bytes, [[maybe_unused]] void* arg) { return false; },
        nullptr);
    } else if constexpr (std::is_same_v<adaptor_type, limiting_resource_adaptor<cuda_mr>>) {
      return std::make_shared<adaptor_type>(upstream, 64_MiB);
    } else if constexpr (std::is_same_v<adaptor_type, owning_wrapper>) {
      return mr::make_owning_wrapper<aligned_resource_adaptor>(std::make_shared<cuda_mr>());
    } else {
      return std::make_shared<adaptor_type>(upstream);
    }
  }
};

TYPED_TEST_SUITE(AdaptorTest, adaptors);

TYPED_TEST(AdaptorTest, NullUpstream)
{
  if constexpr (not std::is_same_v<TypeParam, owning_wrapper>) {
    EXPECT_THROW(this->make_adaptor(nullptr), rmm::logic_error);
  }
}

TYPED_TEST(AdaptorTest, Equality)
{
  EXPECT_TRUE(this->mr->is_equal(*this->mr));

  {
    auto other_mr = this->make_adaptor(&this->cuda);
    EXPECT_TRUE(this->mr->is_equal(*other_mr));
  }

  {
    auto other_mr = aligned_resource_adaptor<rmm::mr::device_memory_resource>{&this->cuda};
    EXPECT_FALSE(this->mr->is_equal(other_mr));
  }
}

TYPED_TEST(AdaptorTest, GetUpstreamResource)
{
  rmm::device_async_resource_ref expected{this->cuda};
  if constexpr (std::is_same_v<TypeParam, owning_wrapper>) {
    EXPECT_TRUE(this->mr->wrapped().get_upstream_resource() == expected);
    EXPECT_TRUE(rmm::mr::is_resource_adaptor<decltype(this->mr->wrapped())>);
  } else {
    EXPECT_TRUE(this->mr->get_upstream_resource() == expected);
    EXPECT_TRUE(rmm::mr::is_resource_adaptor<decltype(*this->mr)>);
  }
}

TYPED_TEST(AdaptorTest, AllocFree)
{
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = this->mr->allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(this->mr->deallocate_sync(ptr, 1024));
}

class LoggingAdaptorTest : public ::testing::Test {
 protected:
  cuda_mr upstream{};
  std::string log_file{"rmm_logging_adaptor_test.txt"};
};

TEST_F(LoggingAdaptorTest, Equality)
{
  logging_resource_adaptor mr1{upstream, log_file};
  logging_resource_adaptor mr2{mr1};

  EXPECT_TRUE(mr1 == mr2);

  logging_resource_adaptor mr3{upstream, "different_file.txt"};
  EXPECT_FALSE(mr1 == mr3);
}

TEST_F(LoggingAdaptorTest, GetUpstreamResource)
{
  logging_resource_adaptor mr{upstream, log_file};
  rmm::device_async_resource_ref expected{upstream};
  EXPECT_TRUE(mr.get_upstream_resource() == expected);
}

TEST_F(LoggingAdaptorTest, AllocateDeallocate)
{
  logging_resource_adaptor mr{upstream, log_file};
  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = mr.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(mr.deallocate_sync(ptr, 1024));
}

TEST_F(LoggingAdaptorTest, SharedOwnership)
{
  logging_resource_adaptor mr1{upstream, log_file};
  logging_resource_adaptor mr2{mr1};

  void* ptr{nullptr};
  EXPECT_NO_THROW(ptr = mr1.allocate_sync(1024));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(mr2.deallocate_sync(ptr, 1024));
}

}  // namespace rmm::test
