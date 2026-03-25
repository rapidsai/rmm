/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/thread_safe_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <mutex>
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
 * @brief Resource that adapts an upstream resource to be thread safe.
 *
 * An instance of this resource can be constructed with an existing, upstream resource in order
 * to satisfy allocation requests. This adaptor wraps allocations and deallocations from the
 * upstream in a mutex lock.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT thread_safe_resource_adaptor
  : public cuda::mr::shared_resource<detail::thread_safe_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::thread_safe_resource_adaptor_impl>;

 public:
  using lock_t = std::lock_guard<std::mutex>;  ///< Type of lock used to synchronize access

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(thread_safe_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new thread safe resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  template <class Upstream,
            std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, thread_safe_resource_adaptor>,
                             int> = 0>
  explicit thread_safe_resource_adaptor(Upstream&& upstream)
    : shared_base(cuda::mr::make_shared_resource<detail::thread_safe_resource_adaptor_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream)}))
  {
  }

  ~thread_safe_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;
};

static_assert(cuda::mr::resource_with<thread_safe_resource_adaptor, cuda::mr::device_accessible>,
              "thread_safe_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
