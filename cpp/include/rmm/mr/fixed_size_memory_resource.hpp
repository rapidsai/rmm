/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/fixed_size_memory_resource_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A `device_memory_resource` which allocates memory blocks of a single fixed size.
 *
 * Supports only allocations of size smaller than the configured block_size.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying pool.
 */
class RMM_EXPORT fixed_size_memory_resource
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::fixed_size_memory_resource_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Equality comparison operator.
   *
   * @param other The other fixed_size_memory_resource to compare against.
   * @return true if both resources share the same underlying state.
   */
  [[nodiscard]] bool operator==(fixed_size_memory_resource const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Inequality comparison operator.
   *
   * @param other The other fixed_size_memory_resource to compare against.
   * @return true if the resources do not share the same underlying state.
   */
  [[nodiscard]] bool operator!=(fixed_size_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

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
   * @param upstream_mr The device_async_resource_ref from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  explicit fixed_size_memory_resource(
    device_async_resource_ref upstream_mr,
    // NOLINTNEXTLINE bugprone-easily-swappable-parameters
    std::size_t block_size            = default_block_size,
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate);

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

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<fixed_size_memory_resource, cuda::mr::device_accessible>,
              "fixed_size_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
