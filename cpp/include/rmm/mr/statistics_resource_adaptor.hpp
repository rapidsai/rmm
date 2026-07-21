/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/statistics_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <utility>

namespace RMM_EXPORT_NAMESPACE {
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
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  explicit statistics_resource_adaptor(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream);

  /**
   * @brief Allocate memory using this resource.
   *
   * @param stream Stream on which to perform the allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory using this resource.
   *
   * @param stream Stream on which to perform deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Allocate memory synchronously using this resource.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory synchronously using this resource.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Compare two resources for equality.
   *
   * @param other The other resource to compare against
   * @return true if the resources compare equal, false otherwise
   */
  [[nodiscard]] bool operator==(statistics_resource_adaptor const& other) const noexcept;
  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other resource to compare against
   * @return true if the resources do not compare equal, false otherwise
   */
  [[nodiscard]] bool operator!=(statistics_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

  ~statistics_resource_adaptor();

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
}  // namespace RMM_EXPORT_NAMESPACE
