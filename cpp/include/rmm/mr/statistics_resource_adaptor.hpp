/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/statistics_resource_adaptor_impl.hpp>
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
 * @brief Resource that uses an upstream resource to allocate memory and tracks allocation
 * statistics (current, peak, total bytes and allocation counts).
 *
 * Supports nested statistics via `push_counters()`/`pop_counters()`. Intended as a debug adaptor.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT statistics_resource_adaptor
  : public cuda::mr::shared_resource<detail::statistics_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::statistics_resource_adaptor_impl>;

 public:
  /// @brief Counter type tracking current, peak, and total bytes or allocations.
  using counter = detail::statistics_resource_adaptor_impl::counter;
  /// @brief Shared-reader lock type used to protect the counter stack.
  using read_lock_t = detail::statistics_resource_adaptor_impl::read_lock_t;
  /// @brief Exclusive-writer lock type used to protect the counter stack.
  using write_lock_t = detail::statistics_resource_adaptor_impl::write_lock_t;

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
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, statistics_resource_adaptor>, int> = 0>
  explicit statistics_resource_adaptor(Upstream&& upstream)
    : shared_base(cuda::mr::make_shared_resource<detail::statistics_resource_adaptor_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream)}))
  {
  }

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
};

static_assert(cuda::mr::resource_with<statistics_resource_adaptor, cuda::mr::device_accessible>,
              "statistics_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
