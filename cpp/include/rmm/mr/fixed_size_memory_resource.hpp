/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/fixed_size_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/std/span>

#include <cstddef>
#include <memory>
#include <vector>

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
   * `upstream`.
   *
   * When the pool of blocks is all allocated, grows the pool by allocating
   * `blocks_to_preallocate` more blocks from `upstream`.
   *
   * @param upstream The resource from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  explicit fixed_size_memory_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
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
};

static_assert(cuda::mr::resource_with<fixed_size_memory_resource, cuda::mr::device_accessible>,
              "fixed_size_memory_resource does not satisfy the cuda::mr::resource concept");

/**
 * @brief RAII handle for an allocation that may span multiple fixed-size blocks from a
 *        `fixed_size_memory_resource`.
 *
 * Returned by `allocate_blocks_async`. When destroyed, all blocks are returned to the
 * memory resource on the same stream used for allocation. Move and copy are disabled to
 * prevent double deallocation. Holds a `fixed_size_memory_resource` (which has shared,
 * refcounted ownership of the underlying pool) so the pool outlives the handle.
 */
class RMM_EXPORT multiple_blocks_allocation {
 public:
  /**
   * @brief Allocate device memory spanning one or more fixed-size blocks, stream-ordered.
   *
   * Use this for allocations larger than a single block. The allocation is ordered on
   * `stream`; deallocation (when the returned handle is destroyed) is also ordered on
   * the same stream. A single event is recorded for the whole allocation, so there is no
   * per-block event overhead.
   *
   * @param mr The `fixed_size_memory_resource` that supplies blocks. Copied by value since
   *        `fixed_size_memory_resource` has refcounted shared ownership.
   * @param size Minimum number of bytes to allocate. Will be rounded up to a multiple of
   *        block size (see `get_block_size()` on `*mr`).
   * @param stream CUDA stream on which the allocation is ordered.
   * @return Unique handle to the allocation; destroys to deallocate. Empty (zero-size)
   *         allocation returns a valid handle with size 0 and no blocks.
   * @throw Any exception from allocating blocks. Blocks successfully taken from the pool
   *        before the failure are returned to the pool on `stream` (same ordering as normal
   *        deallocation).
   */
  [[nodiscard]] static std::unique_ptr<multiple_blocks_allocation> make_async(
    fixed_size_memory_resource mr, std::size_t size, cuda_stream_view stream);

  ~multiple_blocks_allocation();

  multiple_blocks_allocation(multiple_blocks_allocation const&)            = delete;
  multiple_blocks_allocation& operator=(multiple_blocks_allocation const&) = delete;
  multiple_blocks_allocation(multiple_blocks_allocation&&)                 = delete;
  multiple_blocks_allocation& operator=(multiple_blocks_allocation&&)      = delete;

  /**
   * @brief Number of bytes requested for this allocation.
   *
   * @return Requested size in bytes.
   */
  constexpr std::size_t size() const noexcept { return size_; }

  /**
   * @brief Total capacity in bytes (number of blocks × block size).
   *
   * @return Capacity in bytes; always >= size().
   */
  std::size_t capacity() const noexcept { return block_size() * blocks_.size(); }

  /**
   * @brief Size in bytes of each block in this allocation.
   *
   * @return Block size (same as the memory resource's get_block_size()).
   */
  std::size_t block_size() const noexcept { return mr_->get_block_size(); }

  /**
   * @brief Non-owning view of the underlying block pointers.
   *
   * @return Span of device pointers, one per block; each block has size block_size().
   */
  [[nodiscard]] cuda::std::span<std::byte* const> get_blocks() const noexcept
  {
    return {blocks_.data(), blocks_.size()};
  }

  /**
   * @brief Span over the i-th block's bytes.
   *
   * @param i Block index in [0, get_blocks().size()).
   * @return Span of std::byte over the i-th block.
   */
  cuda::std::span<std::byte> operator[](std::size_t i) const
  {
    return {blocks_[i], mr_->get_block_size()};
  }

  /**
   * @brief Span over the i-th block's bytes with bounds checking.
   *
   * @param i Block index.
   * @return Span of std::byte over the i-th block.
   * @throws std::out_of_range if i >= number of blocks.
   */
  cuda::std::span<std::byte> at(std::size_t i) const
  {
    return {blocks_.at(i), mr_->get_block_size()};
  }

  /**
   * @brief Stream on which this allocation is ordered.
   *
   * @return The stream passed to make_async.
   */
  constexpr cuda_stream_view stream() const noexcept { return stream_; }

 private:
  multiple_blocks_allocation(std::size_t size,
                             std::vector<std::byte*> buffers,
                             cuda_stream_view stream,
                             fixed_size_memory_resource mr);

  std::vector<std::byte*> blocks_;
  std::size_t const size_;
  cuda_stream_view stream_;
  fixed_size_memory_resource mr_;
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
