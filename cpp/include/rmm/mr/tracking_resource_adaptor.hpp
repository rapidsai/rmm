/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/tracking_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
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
 * @brief Resource that uses an upstream resource to allocate memory and tracks allocations.
 *
 * Tracks every allocation (size, pointer, and optionally stack trace). Intended as a debug
 * adaptor; should not be used in performance-sensitive code.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT tracking_resource_adaptor
  : public cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl>;

 public:
  /// @brief Allocation info type (pointer, size, optional stack trace).
  using allocation_info = detail::tracking_resource_adaptor_impl::allocation_info;
  /// @brief Shared-reader lock type used to protect the allocations map.
  using read_lock_t = detail::tracking_resource_adaptor_impl::read_lock_t;
  /// @brief Exclusive-writer lock type used to protect the allocations map.
  using write_lock_t = detail::tracking_resource_adaptor_impl::write_lock_t;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(tracking_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a tracking resource adaptor using `upstream` to satisfy allocation requests.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param capture_stacks If true, capture stacks for each allocation.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, tracking_resource_adaptor>, int> = 0>
  tracking_resource_adaptor(Upstream&& upstream, bool capture_stacks = false)
    : shared_base(cuda::mr::make_shared_resource<detail::tracking_resource_adaptor_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream)},
        capture_stacks))
  {
  }

  ~tracking_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Get the outstanding allocations map.
   *
   * @return map of outstanding allocations (pointer → allocation_info)
   */
  [[nodiscard]] std::map<void*, allocation_info> const& get_outstanding_allocations()
    const noexcept;

  /**
   * @brief Query the number of bytes currently allocated.
   *
   * @return std::size_t number of bytes currently allocated
   */
  [[nodiscard]] std::size_t get_allocated_bytes() const noexcept;

  /**
   * @brief Gets a string describing all outstanding allocations (pointer, size, optional stack).
   *
   * @return std::string describing outstanding allocations
   */
  [[nodiscard]] std::string get_outstanding_allocations_str() const;

  /**
   * @brief Log any outstanding allocations via RMM_LOG_DEBUG.
   */
  void log_outstanding_allocations() const;
};

static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>,
              "tracking_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
