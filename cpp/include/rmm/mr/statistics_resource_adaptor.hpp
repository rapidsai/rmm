/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/statistics_resource_adaptor_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that uses an upstream resource to allocate memory and tracks allocation
 * statistics (current, peak, total bytes and allocation counts).
 *
 * Supports nested statistics via `push_counters()`/`pop_counters()`. Intended as a debug adaptor.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT statistics_resource_adaptor
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::statistics_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::statistics_resource_adaptor_impl>;

 public:
  /// @brief Counter type tracking current, peak, and total bytes or allocations.
  using counter = detail::statistics_resource_adaptor_impl::counter;
  /// @brief Shared-reader lock type used to protect the counter stack.
  using read_lock_t = detail::statistics_resource_adaptor_impl::read_lock_t;
  /// @brief Exclusive-writer lock type used to protect the counter stack.
  using write_lock_t = detail::statistics_resource_adaptor_impl::write_lock_t;

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
  [[nodiscard]] bool operator==(statistics_resource_adaptor const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Compare two adaptors for inequality.
   *
   * @param other The other adaptor to compare against.
   * @return true if the adaptors do not share the same underlying impl.
   */
  [[nodiscard]] bool operator!=(statistics_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(statistics_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a statistics resource adaptor using `upstream` to satisfy allocation requests.
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  explicit statistics_resource_adaptor(device_async_resource_ref upstream);

  ~statistics_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Returns a `counter` struct for bytes allocated since construction (or last push).
   *
   * @return counter containing current, peak, and total byte counts
   */
  [[nodiscard]] counter get_bytes_counter() const noexcept;

  /**
   * @brief Returns a `counter` struct for number of allocations since construction (or last push).
   *
   * @return counter containing current, peak, and total allocation counts
   */
  [[nodiscard]] counter get_allocations_counter() const noexcept;

  /**
   * @brief Push a pair of zero counters — new counters start fresh.
   *
   * @return pair of counters (bytes, allocations) from the top _before_ the push
   */
  std::pair<counter, counter> push_counters();

  /**
   * @brief Pop a pair of counters from the stack.
   *
   * @return pair of counters (bytes, allocations) from the top _before_ the pop
   * @throws std::out_of_range if the counter stack has fewer than two entries
   */
  std::pair<counter, counter> pop_counters();

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<statistics_resource_adaptor, cuda::mr::device_accessible>,
              "statistics_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
