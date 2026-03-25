/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/failure_callback_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */

/**
 * @brief A device memory resource that calls a callback function when allocations
 * throw a specified exception type.
 *
 * An instance of this resource must be constructed with an existing, upstream
 * resource in order to satisfy allocation requests.
 *
 * The callback function takes an allocation size and a callback argument and returns
 * a bool representing whether to retry the allocation (true) or re-throw the caught exception
 * (false).
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 *
 * @tparam ExceptionType The type of exception that this adaptor should respond to
 */
template <typename ExceptionType = rmm::out_of_memory>
class failure_callback_resource_adaptor
  : public cuda::mr::shared_resource<
      detail::failure_callback_resource_adaptor_impl<ExceptionType>> {
  using shared_base =
    cuda::mr::shared_resource<detail::failure_callback_resource_adaptor_impl<ExceptionType>>;

 public:
  using exception_type = ExceptionType;  ///< The type of exception this object catches/throws

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(failure_callback_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new `failure_callback_resource_adaptor` using `upstream` to satisfy
   * allocation requests.
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param callback Callback function @see failure_callback_t
   * @param callback_arg Extra argument passed to `callback`
   */
  failure_callback_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                    failure_callback_t callback,
                                    void* callback_arg)
    : shared_base(cuda::mr::make_shared_resource<
                  detail::failure_callback_resource_adaptor_impl<ExceptionType>>(
        std::move(upstream), std::move(callback), callback_arg))
  {
  }

  ~failure_callback_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept
  {
    return this->get().get_upstream_resource();
  }
};

static_assert(
  cuda::mr::resource_with<failure_callback_resource_adaptor<>, cuda::mr::device_accessible>,
  "failure_callback_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
