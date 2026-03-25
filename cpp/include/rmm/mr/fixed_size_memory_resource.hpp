/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/fixed_size_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A memory resource which allocates memory blocks of a single fixed size.
 *
 * Supports only allocations of size smaller than the configured block_size.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying pool.
 */
class RMM_EXPORT fixed_size_memory_resource
  : public cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `fixed_size_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(fixed_size_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  static constexpr std::size_t default_block_size = 1 << 20;  ///< Default allocation block size

  /// The number of blocks that the pool starts out with, and also the number of
  /// blocks by which the pool grows when all of its current blocks are allocated
  static constexpr std::size_t default_blocks_to_preallocate = 128;

  /**
   * @brief Construct a new `fixed_size_memory_resource` that allocates memory from
   * `upstream_mr`.
   *
   * When the pool of blocks is all allocated, grows the pool by allocating
   * `blocks_to_preallocate` more blocks from `upstream_mr`.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream_mr The device_async_resource_ref from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, fixed_size_memory_resource>, int> = 0>
  explicit fixed_size_memory_resource(
    Upstream&& upstream_mr,
    // NOLINTNEXTLINE bugprone-easily-swappable-parameters
    std::size_t block_size            = default_block_size,
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
    : shared_base(cuda::mr::make_shared_resource<detail::fixed_size_memory_resource_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream_mr)},
        block_size,
        blocks_to_preallocate))
  {
  }

  ~fixed_size_memory_resource() = default;

  /**
   * @briefreturn{device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Get the size of blocks allocated by this memory resource.
   *
   * @return std::size_t size in bytes of allocated blocks.
   */
  [[nodiscard]] std::size_t get_block_size() const noexcept;
};

static_assert(cuda::mr::resource_with<fixed_size_memory_resource, cuda::mr::device_accessible>,
              "fixed_size_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
