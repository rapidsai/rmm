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

#include <unordered_set>

namespace rmm {
namespace mr {
namespace detail {
namespace arena {

struct block {
  static constexpr size_t superblock_size = 4194304;  // 4 MiB

  void* pointer{};
  size_t size;
  bool is_head;

  bool is_superblock() const { return is_head && size == superblock_size; }

  /// Returns true if this block is valid (non-null), false otherwise
  bool is_valid() const { return pointer != nullptr; }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
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
  bool fits(size_t sz) const { return size >= sz; }

  std::pair<void*, block> split(size_t sz) const
  {
    if (size > sz) {
      return {pointer, {increment(pointer, sz), sz, false}};
    } else {
      return {pointer, {}};
    }
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param b block to merge
   * @return block The merged block
   */
  block merge(block const& b) const
  {
    assert(is_contiguous_before(b));
    return {pointer, size + b.size, is_head};
  }

  static void* increment(void* ptr, size_t size)
  {
    return static_cast<void*>(static_cast<char*>(ptr) + size);
  }

  /// Used by std::set to compare blocks.
  bool operator<(const block& b) const { return pointer < b.pointer; }

  /// Used by std::unordered_set to compare blocks.
  bool operator==(const block& b) const { return pointer == b.pointer; }
};

/// Used by std::unordered_set to hash blocks.
struct block_hash {
  size_t operator()(const block& b) const { return std::hash<void*>()(b.pointer); }
};

template <typename Upstream>
class arena {
 public:
  explicit arena(Upstream* upstream_mr) : upstream_mr_{upstream_mr} {}

  // Disable copy (and move) semantics.
  arena(const arena&) = delete;
  arena& operator=(const arena&) = delete;

  static constexpr size_t maximum_allocation_size() { return block::superblock_size; }

  void* allocate(std::size_t bytes, cudaStream_t stream)
  {
    lock_guard lock(mtx_);
    auto const b = get_block(bytes);
    auto split   = allocate_from_block(b, bytes);
    if (split.second.is_valid()) free_blocks.insert(split.second);
    return split.first;
  }

  bool deallocate(void* p, std::size_t bytes, cudaStream_t stream)
  {
    lock_guard lock(mtx_);
    auto b = free_block(p, bytes);
    if (b.is_valid()) { insert_and_coalesce(b); }
    if (free_superblocks.size() > 1) { shrink_arena(stream); }
    return b.is_valid();
  }

  bool deallocate(void* p, std::size_t bytes)
  {
    lock_guard lock(mtx_);
    auto b = free_block(p, bytes);
    if (b.is_valid()) { insert_and_coalesce(b); }
    return b.is_valid();
  }

 private:
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @return block A block of memory of at least `size` bytes
   */
  block get_block(size_t size)
  {
    // Find first fit block.
    auto const iter = std::find_if(
      free_blocks.cbegin(), free_blocks.cend(), [size](block const& b) { return b.fits(size); });

    if (iter != free_blocks.cend()) {
      // Remove the block from the free_list and return it.
      block const found = *iter;
      free_blocks.erase(iter);
      return found;
    }

    // No larger blocks available, so grow the arena and create a superblock.
    return expand_arena();
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
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the arena.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  std::pair<void*, block> allocate_from_block(block const& b, size_t size)
  {
    block const alloc{b.pointer, size, b.is_head};
    allocated_blocks.insert(alloc);
    return b.split(size);
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the arena.
   */
  block free_block(void* p, size_t size) noexcept
  {
    block b{p};
    auto const i = allocated_blocks.find(b);

    // The pointer may be allocated in another arena.
    if (i == allocated_blocks.end()) { return {}; }

    auto found = *i;
    assert(found.size == size);
    allocated_blocks.erase(i);

    return found;
  }

  void insert_and_coalesce(block b)
  {
    // Find the right place (in ascending ptr order) to insert the block.
    auto const next     = free_blocks.lower_bound(b);
    auto const previous = (next == free_blocks.cbegin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks or insert the new block if it can't be coalesced.
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next = (next != free_blocks.cend()) && b.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      b = previous->merge(b).merge(*next);
      free_blocks.erase(previous);
      free_blocks.erase(next);
    } else if (merge_prev) {
      b = previous->merge(b);
      free_blocks.erase(previous);
    } else if (merge_next) {
      b = b.merge(*next);
      free_blocks.erase(next);
    }
    free_blocks.insert(b);
    if (b.is_superblock()) { free_superblocks.insert(b); }
  }

  void shrink_arena(cudaStream_t stream)
  {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    // Always keep the first superblock.
    for (auto it = std::next(free_superblocks.begin()); it != free_superblocks.end(); ++it) {
      auto b = *it;
      upstream_mr_->deallocate(b.pointer, b.size, cudaStreamLegacy);
      free_superblocks.erase(it--);
      free_blocks.erase(b);
    }
  }

  Upstream* upstream_mr_;
  std::set<block> free_blocks;
  std::set<block> free_superblocks;
  std::unordered_set<block, block_hash> allocated_blocks;
  mutable std::mutex mtx_;
};

}  // namespace arena
}  // namespace detail
}  // namespace mr
}  // namespace rmm
