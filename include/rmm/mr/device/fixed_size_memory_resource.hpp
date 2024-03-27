/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/detail/fixed_size_free_list.hpp>
#include <rmm/mr/device/detail/stream_ordered_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <list>
#include <map>
#include <utility>
#include <vector>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
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
   * @brief Construct a new `fixed_size_memory_resource` that allocates memory from
   * `upstream_resource`.
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
    std::size_t block_size            = default_block_size,
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
    : upstream_mr_{upstream_mr},
      block_size_{rmm::align_up(block_size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      upstream_chunk_size_{block_size * blocks_to_preallocate}
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
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_mr_;
  }

  /**
   * @briefreturn{Upstream* to the upstream memory resource}
   */
  [[nodiscard]] Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Get the size of blocks allocated by this memory resource.
   *
   * @return std::size_t size in bytes of allocated blocks.
   */
  [[nodiscard]] std::size_t get_block_size() const noexcept { return block_size_; }

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
    void* ptr = get_upstream_resource().allocate_async(upstream_chunk_size_, stream);
    block_type block{ptr};
    upstream_blocks_.push_back(block);

    auto num_blocks = upstream_chunk_size_ / block_size_;

    auto block_gen = [ptr, this](int index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return block_type{static_cast<char*>(ptr) + index * block_size_};
    };
    auto first =
      thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), block_gen);
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
  split_block allocate_from_block(block_type const& block, std::size_t size)
  {
    return {block, block_type{nullptr}};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* ptr, std::size_t size) noexcept
  {
    // Deallocating a fixed-size block just inserts it in the free list, which is
    // handled by the parent class
    RMM_LOGGING_ASSERT(rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT) <= block_size_);
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
      get_upstream_resource().deallocate(block.pointer(), upstream_chunk_size_);
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
  Upstream* upstream_mr_;  // The resource from which to allocate new blocks

  std::size_t const block_size_;           // size of blocks this MR allocates
  std::size_t const upstream_chunk_size_;  // size of chunks allocated from heap MR

  // blocks allocated from heap: so they can be easily freed
  std::vector<block_type> upstream_blocks_;
};

/** @} */  // end of group
}  // namespace rmm::mr
