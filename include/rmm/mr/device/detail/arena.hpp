/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <include/rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <mutex>
#include <set>
#include <unordered_map>

namespace rmm {
namespace mr {
namespace detail {
namespace arena {

/**
 * @brief Represents a chunk of memory that can be allocated and deallocated.
 *
 * A fixed-sized block obtained from the upstream memory resource is called a "superblock". Only a
 * superblock can be returned to upstream.
 */
struct block {
  static constexpr std::size_t superblock_size = 8388608;  ///> Size of a superblock (8 MiB)

  void* pointer{};      ///< Raw memory pointer
  std::size_t size{};   ///< Size in bytes
  bool is_head{false};  ///< Indicates whether pointer was allocated from upstream

  /// Returns true if this block is valid (non-null), false otherwise.
  bool is_valid() const { return pointer != nullptr; }

  /// Returns true if this block is a superblock, false otherwise.
  bool is_superblock() const { return is_head && size == superblock_size; }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this block's `pointer` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  bool is_contiguous_before(block const& b) const
  {
    return increment(pointer, size) == b.pointer and not(b.is_head);
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes
   */
  bool fits(std::size_t sz) const { return size >= sz; }

  /**
   * @brief Split this block into a pointer and a remainder block.
   *
   * @param sz The size in bytes of the first block.
   * @return std::pair<block, block> A pair of blocks split by sz
   */
  std::pair<block, block> split(std::size_t sz) const
  {
    assert(size >= sz);
    if (size > sz) {
      return {{pointer, sz, is_head}, {increment(pointer, sz), size - sz, false}};
    } else {
      return {*this, {}};
    }
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same superblock.
   *
   * @param b block to merge
   * @return block The merged block
   */
  block merge(block const& b) const
  {
    assert(is_contiguous_before(b));
    return {pointer, size + b.size, is_head};
  }

  /**
   * @brief Increment the given pointer by the given size.
   *
   * @param pointer the pointer to increment
   * @param size the size to add to the pointer
   * @return void* The resulting pointer
   */
  static void* increment(void* pointer, std::size_t size)
  {
    return static_cast<void*>(static_cast<char*>(pointer) + size);
  }

  /// Used by std::set to compare blocks.
  bool operator<(block const& b) const { return pointer < b.pointer; }
};

/**
 * @brief An arena for allocating memory for a thread.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena {
 public:
  /**
   * @brief Construct an `arena`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate memory for the arena.
   */
  explicit arena(Upstream* upstream_mr) : upstream_mr_{upstream_mr}
  {
    RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");
  }

  // Disable copy (and move) semantics.
  arena(const arena&) = delete;
  arena& operator=(const arena&) = delete;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size in bytes of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes)
  {
    lock_guard lock(mtx_);
    auto const b     = get_block(bytes);
    auto const alloc = allocate_from_block(b, bytes);
    return alloc.pointer;
  }

  /**
   * @brief Deallocate memory pointed to by `p`, and possibly return superblocks to upstream.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   * @return bool if the allocation is found
   */
  bool deallocate(void* p, std::size_t bytes, cudaStream_t stream)
  {
    lock_guard lock(mtx_);
    bool found = do_deallocate(p, bytes);
    shrink_arena(stream);
    return found;
  }

  /**
   * @brief Deallocate memory pointed to by `p`, keeping all free superblocks.
   *
   * This is done when deallocating across arenas. Since we don't have access to the CUDA stream
   * associated with this arena, it's not safe to return superblocks.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @return bool if the allocation is found
   */
  bool deallocate(void* p, std::size_t bytes)
  {
    lock_guard lock(mtx_);
    return do_deallocate(p, bytes);
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @return bool if the allocation is found
   */
  bool do_deallocate(void* p, std::size_t bytes)
  {
    auto b = free_block(p, bytes);
    coalesce_block(b);
    return b.is_valid();
  }

  /**
   * @brief Does an arena handle blocks of this size?
   *
   * @param size The size in bytes to check
   * @return true if blocks of this size are handled by an arena
   */
  static bool handles_size(std::size_t size) { return size <= maximum_allocation_size; }

 private:
  /// The maximum allocation size handled by arenas (4 MiB).
  static constexpr std::size_t maximum_allocation_size = block::superblock_size / 2;  // 4 MiB

  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate
   * @return block A block of memory of at least `size` bytes
   */
  block get_block(std::size_t size)
  {
    // Find the first-fit free block.
    block b = first_fit(size);
    if (b.is_valid()) return b;

    // No free blocks available, use the first free superblock.
    b = first_superblock();
    if (b.is_valid()) return b;

    // No existing larger blocks available, so grow the arena and obtain a superblock.
    return expand_arena();
  }

  /**
   * @brief Get the first free block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate
   * @return block A block of memory of at least `size` bytes
   */
  block first_fit(std::size_t size)
  {
    auto const iter = std::find_if(
      free_blocks_.cbegin(), free_blocks_.cend(), [size](block const& b) { return b.fits(size); });

    if (iter == free_blocks_.cend()) {
      return {};
    } else {
      // Remove the block from the free_list and return it.
      block const found = *iter;
      assert(found.fits(size));
      free_blocks_.erase(iter);
      return found;
    }
  }

  /**
   * @brief Get the first free superblock.
   *
   * @return block A superblock
   */
  block first_superblock()
  {
    if (free_superblocks_.empty()) {
      return {};
    } else {
      auto const iter   = free_superblocks_.cbegin();
      block const found = *iter;
      assert(found.is_superblock());
      free_superblocks_.erase(iter);
      return found;
    }
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a superblock.
   *
   * @return block a superblock
   */
  block expand_arena()
  {
    auto size = block::superblock_size;
    return {upstream_mr_->allocate(size, cudaStreamLegacy), size, true};
  }

  /**
   * @brief Splits block `b` if necessary to return the block allocated.
   *
   * If the block is split, the remainder is returned to the arena.
   *
   * @param b The block to allocate from
   * @param size The size in bytes of the requested allocation
   * @return block The allocated block
   */
  block allocate_from_block(block const& b, std::size_t size)
  {
    assert(b.fits(size));
    auto split     = b.split(size);
    auto alloc     = split.first;
    auto remainder = split.second;
    allocated_blocks_.emplace(alloc.pointer, alloc);
    if (remainder.is_valid()) free_blocks_.insert(remainder);
    return alloc;
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the arena.
   */
  block free_block(void* p, std::size_t size) noexcept
  {
    auto const i = allocated_blocks_.find(p);

    // The pointer may be allocated in another arena.
    if (i == allocated_blocks_.end()) { return {}; }

    auto found = i->second;
    assert(found.size == size);
    allocated_blocks_.erase(i);

    return found;
  }

  /**
   * @brief Coalesce the given block with other free blocks.
   *
   * @param b The block to coalesce
   */
  void coalesce_block(block b)
  {
    if (!b.is_valid()) return;

    // Find the right place (in ascending address order) to insert the block.
    auto const next     = free_blocks_.lower_bound(b);
    auto const previous = (next == free_blocks_.cbegin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks.
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next = (next != free_blocks_.cend()) && b.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      b = previous->merge(b).merge(*next);
      free_blocks_.erase(previous);
      free_blocks_.erase(next);
    } else if (merge_prev) {
      b = previous->merge(b);
      free_blocks_.erase(previous);
    } else if (merge_next) {
      b = b.merge(*next);
      free_blocks_.erase(next);
    }

    // Keep track of the coalesced free block. Free superblocks are tracked separately.
    if (b.is_superblock()) {
      free_superblocks_.insert(b);
    } else {
      free_blocks_.insert(b);
    }
  }

  /**
   * @brief Shrink this arena by returning free superblocks to upstream.
   *
   * @param stream Stream on which to perform shrinking
   */
  void shrink_arena(cudaStream_t stream)
  {
    // Don't shrink if no free superblocks, or no free blocks and only one superblock is left (to
    // avoid thrashing).
    if (free_superblocks_.empty() || free_blocks_.empty() && free_superblocks_.size() <= 1) return;

    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    // Keep one superblock if no free blocks are available.
    for (auto it = free_blocks_.empty() ? std::next(free_superblocks_.cbegin())
                                        : free_superblocks_.cbegin();
         it != free_superblocks_.cend();
         ++it) {
      auto b = *it;
      assert(b.is_superblock());
      upstream_mr_->deallocate(b.pointer, b.size, cudaStreamLegacy);
      free_superblocks_.erase(it--);
    }
  }

  Upstream* upstream_mr_;
  std::set<block> free_blocks_;
  std::set<block> free_superblocks_;
  std::unordered_map<void*, block> allocated_blocks_;
  mutable std::mutex mtx_;
};

}  // namespace arena
}  // namespace detail
}  // namespace mr
}  // namespace rmm
