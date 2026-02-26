/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/tracking_resource_adaptor_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <map>
#include <memory>
#include <string>

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
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::tracking_resource_adaptor_impl>;

 public:
  /// @brief Allocation info type (pointer, size, optional stack trace).
  using allocation_info = detail::tracking_resource_adaptor_impl::allocation_info;
  /// @brief Shared-reader lock type used to protect the allocations map.
  using read_lock_t = detail::tracking_resource_adaptor_impl::read_lock_t;
  /// @brief Exclusive-writer lock type used to protect the allocations map.
  using write_lock_t = detail::tracking_resource_adaptor_impl::write_lock_t;

  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Compare two adaptors for equality (shared-impl identity).
   *
   * @param other The other adaptor to compare against.
   * @return true if both adaptors share the same underlying impl.
   */
  [[nodiscard]] bool operator==(tracking_resource_adaptor const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Compare two adaptors for inequality.
   *
   * @param other The other adaptor to compare against.
   * @return true if the adaptors do not share the same underlying impl.
   */
  [[nodiscard]] bool operator!=(tracking_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

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
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param capture_stacks If true, capture stacks for each allocation.
   */
  tracking_resource_adaptor(device_async_resource_ref upstream, bool capture_stacks = false);

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

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>,
              "tracking_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
