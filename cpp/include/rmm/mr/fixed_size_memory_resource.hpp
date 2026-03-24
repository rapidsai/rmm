/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/mr/detail/fixed_size_free_list.hpp>
#include <rmm/mr/detail/stream_ordered_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

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
 */
template <typename Upstream>
class fixed_size_memory_resource
  : public detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                  detail::fixed_size_free_list> {
 public:
  friend class detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                      detail::fixed_size_free_list>;

  static constexpr std::size_t default_block_size = 1 << 20;  ///< Default allocation block size

  /// The number of blocks that the pool starts out with, and also the number of
  /// blocks by which the pool grows when all of its current blocks are allocated
  static constexpr std::size_t default_blocks_to_preallocate = 128;

  /**
   * @brief RAII handle for an allocation that may span multiple fixed-size blocks.
   *
   * Returned by `allocate_blocks_async`. When destroyed, all blocks are returned to the
   * memory resource on the same stream used for allocation. Move and copy are disabled to
   * prevent double deallocation.
   */
  struct multiple_blocks_allocation {
    friend class fixed_size_memory_resource<Upstream>;

    ~multiple_blocks_allocation()
    {
      if (mr_ && !blocks_.empty()) { mr_->deallocate_blocks_async(std::move(blocks_), stream_); }
    }

    // Disable copy to prevent double deallocation
    multiple_blocks_allocation(const multiple_blocks_allocation&)            = delete;
    multiple_blocks_allocation& operator=(const multiple_blocks_allocation&) = delete;
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
    constexpr std::size_t capacity() const noexcept { return block_size() * blocks_.size(); }

    /**
     * @brief Size in bytes of each block in this allocation.
     *
     * @return Block size (same as the memory resource's get_block_size()).
     */
    constexpr std::size_t block_size() const noexcept { return mr_->get_block_size(); }

    /**
     * @brief Non-owning view of the underlying block pointers.
     *
     * @return Span of device pointers, one per block; each block has size block_size().
     */
    cuda::std::span<std::byte* const> get_blocks() const noexcept
    {
      return cuda::std::span<std::byte* const>(blocks_.data(), blocks_.size());
    }

    /**
     * @brief Span over the i-th block's bytes.
     *
     * @param i Block index in [0, get_blocks().size()).
     * @return Span of std::byte over the i-th block.
     */
    cuda::std::span<std::byte> operator[](std::size_t i) const
    {
      return cuda::std::span<std::byte>{blocks_[i], mr_->get_block_size()};
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
      return cuda::std::span<std::byte>{blocks_.at(i), mr_->get_block_size()};
    }

    /**
     * @brief Stream on which this allocation is ordered.
     *
     * @return The stream passed to allocate_blocks_async.
     */
    constexpr cuda_stream_view stream() const noexcept { return stream_; }

   private:
    explicit multiple_blocks_allocation(std::size_t size,
                                        std::vector<std::byte*> buffers,
                                        cuda_stream_view stream,
                                        fixed_size_memory_resource<Upstream>* m)
      : blocks_(std::move(buffers)), size_(size), stream_(stream), mr_(m)
    {
      RMM_LOGGING_ASSERT(size_ <= mr_->get_block_size() * blocks_.size());
      RMM_LOGGING_ASSERT(blocks_.size() == cuda::ceil_div(size_, mr_->get_block_size()));
    }

    std::vector<std::byte*> blocks_;
    const std::size_t size_;
    cuda_stream_view stream_;
    fixed_size_memory_resource<Upstream>* mr_;
  };

  /**
   * @brief Allocate device memory spanning one or more fixed-size blocks, stream-ordered.
   *
   * Use this for allocations larger than a single block. The allocation is ordered on
   * `stream`; deallocation (when the returned handle is destroyed) is also ordered on
   * the same stream. A single event is recorded for the whole allocation, so there is no
   * per-block event overhead.
   *
   * @param size Minimum number of bytes to allocate. Will be rounded up to a multiple of
   *        block size (see get_block_size()).
   * @param stream CUDA stream on which the allocation is ordered.
   * @return Unique handle to the allocation; destroys to deallocate. Empty (zero-size)
   *         allocation returns a valid handle with size 0 and no blocks.
   */
  std::unique_ptr<multiple_blocks_allocation> allocate_blocks_async(std::size_t size,
                                                                    cuda_stream_view stream)
  {
    if (size == 0) { return std::make_unique<multiple_blocks_allocation>(0, {}, stream, this); }

    lock_guard lock(this->get_mutex());

    auto stream_event            = this->get_event(stream);
    std::size_t const num_blocks = cuda::ceil_div(size, get_block_size());
    std::vector<std::byte*> blocks;
    blocks.resize(num_blocks);
    cuda::std::generate_n(blocks.begin(), num_blocks, [this, &stream_event]() {
      return static_cast<std::byte*>(this->get_block(get_block_size(), stream_event).pointer());
    });

    return std::make_unique<multiple_blocks_allocation>(size, std::move(blocks), stream, this);
  }

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
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
    : upstream_mr_{upstream_mr},
      block_size_{align_up(block_size, CUDA_ALLOCATION_ALIGNMENT)},
      upstream_chunk_size_{block_size_ * blocks_to_preallocate}
  {
    // allocate initial blocks and insert into free list
    this->insert_blocks(std::move(blocks_from_upstream(cuda_stream_legacy)), cuda_stream_legacy);
  }

  /**
   * @brief Construct a new `fixed_size_memory_resource` that allocates memory from
   * `upstream_mr`.
   *
   * When the pool of blocks is all allocated, grows the pool by allocating
   * `blocks_to_preallocate` more blocks from `upstream_mr`.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  explicit fixed_size_memory_resource(
    Upstream* upstream_mr,
    // NOLINTNEXTLINE bugprone-easily-swappable-parameters
    std::size_t block_size            = default_block_size,
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
    : upstream_mr_{to_device_async_resource_ref_checked(upstream_mr)},
      block_size_{align_up(block_size, CUDA_ALLOCATION_ALIGNMENT)},
      upstream_chunk_size_{block_size_ * blocks_to_preallocate}
  {
    // allocate initial blocks and insert into free list
    this->insert_blocks(std::move(blocks_from_upstream(cuda_stream_legacy)), cuda_stream_legacy);
  }

  /**
   * @brief Destroy the `fixed_size_memory_resource` and free all memory allocated from upstream.
   *
   */
  ~fixed_size_memory_resource() override { release(); }

  fixed_size_memory_resource()                                             = delete;
  fixed_size_memory_resource(fixed_size_memory_resource const&)            = delete;
  fixed_size_memory_resource(fixed_size_memory_resource&&)                 = delete;
  fixed_size_memory_resource& operator=(fixed_size_memory_resource const&) = delete;
  fixed_size_memory_resource& operator=(fixed_size_memory_resource&&)      = delete;

  /**
   * @briefreturn{device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_mr_;
  }

  /**
   * @brief Get the size of blocks allocated by this memory resource.
   *
   * @return std::size_t size in bytes of allocated blocks.
   */
  [[nodiscard]] constexpr std::size_t get_block_size() const noexcept { return block_size_; }

 protected:
  using free_list  = detail::fixed_size_free_list;  ///< The free list type
  using block_type = free_list::block_type;         ///< The type of block managed by the free list
  using typename detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                        detail::fixed_size_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;  ///< Type of lock used to synchronize access

  /**
   * @brief Get the (fixed) size of allocations supported by this memory resource
   *
   * @return std::size_t The (fixed) maximum size of a single allocation supported by this memory
   * resource
   */
  [[nodiscard]] std::size_t get_maximum_allocation_size() const { return get_block_size(); }

  /**
   * @brief Allocate a block from upstream to supply the suballocation pool.
   *
   * Note typically the allocated size will be larger than requested, and is based on the growth
   * strategy (see `size_to_grow()`).
   *
   * @param size The minimum size to allocate
   * @param blocks The set of blocks from which to allocate
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)
  {
    blocks.insert(std::move(blocks_from_upstream(stream)));
    return blocks.get_block(size);
  }

  /**
   * @brief Allocate blocks from upstream to expand the suballocation pool.
   *
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  free_list blocks_from_upstream(cuda_stream_view stream)
  {
    void* ptr = get_upstream_resource().allocate(stream, upstream_chunk_size_);
    block_type block{ptr};
    upstream_blocks_.push_back(block);

    auto num_blocks = upstream_chunk_size_ / block_size_;

    auto block_gen = [ptr, this](std::size_t index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return block_type{static_cast<char*>(ptr) + index * block_size_};
    };
    auto first =
      cuda::make_transform_iterator(cuda::make_counting_iterator(std::size_t{0}), block_gen);
    return free_list(first, first + num_blocks);
  }

  /**
   * @brief Splits block if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param block The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& block, [[maybe_unused]] std::size_t size)
  {
    return {block, block_type{nullptr}};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `ptr`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* ptr, [[maybe_unused]] std::size_t size) noexcept
  {
    // Deallocating a fixed-size block just inserts it in the free list, which is
    // handled by the parent class
    RMM_LOGGING_ASSERT(align_up(size, CUDA_ALLOCATION_ALIGNMENT) <= block_size_);
    return block_type{ptr};
  }

  /**
   * @brief free all memory allocated using the upstream resource.
   *
   */
  void release()
  {
    lock_guard lock(this->get_mutex());

    for (auto block : upstream_blocks_) {
      get_upstream_resource().deallocate_sync(block.pointer(), upstream_chunk_size_);
    }
    upstream_blocks_.clear();
  }

#ifdef RMM_DEBUG_PRINT
  void print()
  {
    lock_guard lock(this->get_mutex());

    auto const [free, total] = rmm::available_device_memory();
    std::cout << "GPU free memory: " << free << " total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto blocks : upstream_blocks_) {
      blocks.print();
      upstream_total += upstream_chunk_size_;
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    this->print_free_blocks();
  }
#endif

  /**
   * @brief Get the largest available block size and total free size in the specified free list
   *
   * This is intended only for debugging
   *
   * @param blocks The free list from which to return the summary
   * @return std::pair<std::size_t, std::size_t> Pair of largest available block, total free size
   */
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks)
  {
    return blocks.is_empty() ? std::make_pair(std::size_t{0}, std::size_t{0})
                             : std::make_pair(block_size_, blocks.size() * block_size_);
  }

 private:
  void deallocate_blocks_async(std::vector<std::byte*>&& blocks, cuda_stream_view stream)
  {
    if (blocks.empty()) { return; }

    lock_guard lock(this->get_mutex());

    free_list blocks_free_list;
    cuda::std::ranges::for_each(blocks, [this, &blocks_free_list](std::byte* ptr) {
      blocks_free_list.insert(this->free_block(ptr, get_block_size()));
    });

    auto stream_event = this->get_event(stream);
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, stream.value()));
    this->insert_blocks(std::move(blocks_free_list), stream);
  }

  device_async_resource_ref upstream_mr_;  // The resource from which to allocate new blocks

  std::size_t block_size_;           // size of blocks this MR allocates
  std::size_t upstream_chunk_size_;  // size of chunks allocated from heap MR

  // blocks allocated from heap: so they can be easily freed
  std::vector<block_type> upstream_blocks_;
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
