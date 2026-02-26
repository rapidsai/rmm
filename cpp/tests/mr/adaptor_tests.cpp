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
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/thread_safe_resource_adaptor.hpp>
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
using owning_wrapper = rmm::mr::owning_wrapper<limiting_resource_adaptor<cuda_mr>, cuda_mr>;

// explicit instantiations for test coverage purposes
template class rmm::mr::failure_callback_resource_adaptor<cuda_mr>;
template class rmm::mr::limiting_resource_adaptor<cuda_mr>;
template class rmm::mr::thread_safe_resource_adaptor<cuda_mr>;

namespace rmm::test {

using adaptors = ::testing::Types<failure_callback_resource_adaptor<cuda_mr>,
                                  limiting_resource_adaptor<cuda_mr>,
                                  owning_wrapper,
                                  thread_safe_resource_adaptor<cuda_mr>>;

// static property checks
static_assert(
  rmm::detail::polyfill::resource_with<rmm::mr::failure_callback_resource_adaptor<cuda_mr>,
                                       cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::limiting_resource_adaptor<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::owning_wrapper<cuda_mr>,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::resource_with<rmm::mr::thread_safe_resource_adaptor<cuda_mr>,
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
      return mr::make_owning_wrapper<limiting_resource_adaptor>(std::make_shared<cuda_mr>(),
                                                                64_MiB);
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
    auto other_mr = aligned_resource_adaptor{this->cuda};
    EXPECT_FALSE(this->mr->is_equal(other_mr));
  }
}

TYPED_TEST(AdaptorTest, GetUpstreamResource)
{
  rmm::device_async_resource_ref expected{this->cuda};
  if constexpr (std::is_same_v<TypeParam, owning_wrapper>) {
    EXPECT_EQ(this->mr->wrapped().get_upstream_resource(), expected);
    EXPECT_TRUE(rmm::mr::is_resource_adaptor<decltype(this->mr->wrapped())>);
  } else {
    EXPECT_EQ(this->mr->get_upstream_resource(), expected);
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

}  // namespace rmm::test
