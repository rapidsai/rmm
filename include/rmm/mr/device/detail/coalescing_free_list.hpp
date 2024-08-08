/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/detail/free_list.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <list>

namespace rmm::mr::detail {

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
struct block : public block_base {
  block() = default;
  block(char* ptr, std::size_t size, bool is_head)
    : block_base{ptr}, size_bytes{size}, head{is_head}
  {
  }

  /**
   * @brief Returns the pointer to the memory represented by this block.
   *
   * @return the pointer to the memory represented by this block.
   */
  [[nodiscard]] inline char* pointer() const { return static_cast<char*>(ptr); }

  /**
   * @brief Returns the size of the memory represented by this block.
   *
   * @return the size in bytes of the memory represented by this block.
   */
  [[nodiscard]] inline std::size_t size() const { return size_bytes; }

  /**
   * @brief Returns whether this block is the start of an allocation from an upstream allocator.
   *
   * A block `b` may not be coalesced with a preceding contiguous block `a` if `b.is_head == true`.
   *
   * @return true if this block is the start of an allocation from an upstream allocator.
   */
  [[nodiscard]] inline bool is_head() const { return head; }

  /**
   * @brief Comparison operator to enable comparing blocks and storing in ordered containers.
   *
   * Orders by ptr address.

   * @param rhs
   * @return true if this block's ptr is < than `rhs` block pointer.
   * @return false if this block's ptr is >= than `rhs` block pointer.
   */
  inline bool operator<(block const& rhs) const noexcept { return pointer() < rhs.pointer(); };

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param blk block to merge
   * @return The merged block
   */
  [[nodiscard]] inline block merge(block const& blk) const noexcept
  {
    assert(is_contiguous_before(blk));
    return {pointer(), size() + blk.size(), is_head()};
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param blk The block to check for contiguity.
   * @return Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
             false otherwise.
   */
  [[nodiscard]] inline bool is_contiguous_before(block const& blk) const noexcept
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return (pointer() + size() == blk.ptr) and not(blk.is_head());
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param bytes The size in bytes to check for fit.
   * @return true if this block is at least `bytes` bytes
   */
  [[nodiscard]] inline bool fits(std::size_t bytes) const noexcept { return size() >= bytes; }

  /**
   * @brief Is this block a better fit for `sz` bytes than block `b`?
   *
   * @param bytes The size in bytes to check for best fit.
   * @param blk The other block to check for fit.
   * @return true If this block is a tighter fit for `bytes` bytes than block `blk`.
   * @return false If this block does not fit `bytes` bytes or `blk` is a tighter fit.
   */
  [[nodiscard]] inline bool is_better_fit(std::size_t bytes, block const& blk) const noexcept
  {
    return fits(bytes) && (size() < blk.size() || blk.size() < bytes);
  }

#ifdef RMM_DEBUG_PRINT
  /**
   * @brief Print this block. For debugging.
   */
  inline void print() const
  {
    std::cout << fmt::format("{} {} B", fmt::ptr(pointer()), size()) << std::endl;
  }
#endif

 private:
  std::size_t size_bytes{};  ///< Size in bytes
  bool head{};               ///< Indicates whether ptr was allocated from the heap
};

#ifdef RMM_DEBUG_PRINT
/// Print block on an ostream
inline std::ostream& operator<<(std::ostream& out, const block& blk)
{
  out << fmt::format("{} {} B\n", fmt::ptr(blk.pointer()), blk.size());
  return out;
}
#endif

/**
 * @brief Comparator for block types based on pointer address.
 *
 * This comparator allows searching associative containers of blocks by pointer rather than
 * having to search by the contained type. Saves potentially error-prone temporary construction of
 * a block when you just want to search by pointer.
 */
template <typename block_type>
struct compare_blocks {
  // is_transparent (C++14 feature) allows search key type for set<block_type>::find()
  using is_transparent = void;

  bool operator()(block_type const& lhs, block_type const& rhs) const { return lhs < rhs; }
  bool operator()(char const* ptr, block_type const& rhs) const { return ptr < rhs.pointer(); }
  bool operator()(block_type const& lhs, char const* ptr) const { return lhs.pointer() < ptr; };
};

/**
 * @brief An ordered list of free memory blocks that coalesces contiguous blocks on insertion.
 *
 * @tparam list_type the type of the internal list data structure.
 */
struct coalescing_free_list : free_list<block> {
  coalescing_free_list()           = default;
  ~coalescing_free_list() override = default;

  coalescing_free_list(coalescing_free_list const&)            = delete;
  coalescing_free_list& operator=(coalescing_free_list const&) = delete;
  coalescing_free_list(coalescing_free_list&&)                 = delete;
  coalescing_free_list& operator=(coalescing_free_list&&)      = delete;

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   *
   * @param b The block to insert.
   */
  void insert(block_type const& block)
  {
    if (is_empty()) {
      free_list::insert(cend(), block);
      return;
    }

    // Find the right place (in ascending ptr order) to insert the block
    // Can't use binary_search because it's a linked list and will be quadratic
    auto const next =
      std::find_if(begin(), end(), [block](block_type const& blk) { return block < blk; });
    auto const previous = (next == cbegin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks or insert the new block if it can't be coalesced
    bool const merge_prev = previous->is_contiguous_before(block);
    bool const merge_next = (next != cend()) && block.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      *previous = previous->merge(block).merge(*next);
      erase(next);
    } else if (merge_prev) {
      *previous = previous->merge(block);
    } else if (merge_next) {
      *next = block.merge(*next);
    } else {
      free_list::insert(next, block);  // cannot be coalesced, just insert
    }
  }

  /**
   * @brief Moves blocks from free_list `other` into this free_list in their correct order,
   *        coalescing them with their preceding and following blocks if they are contiguous.
   *
   * @tparam InputIt iterator type
   * @param other free_list of blocks to insert
   */
  void insert(free_list&& other)
  {
    using std::make_move_iterator;
    auto inserter = [this](block_type&& block) { this->insert(block); };
    std::for_each(make_move_iterator(other.begin()), make_move_iterator(other.end()), inserter);
  }

  /**
   * @brief Finds the smallest block in the `free_list` large enough to fit `size` bytes.
   *
   * This is a "best fit" search.
   *
   * @param size The size in bytes of the desired block.
   * @return A block large enough to store `size` bytes.
   */
  block_type get_block(std::size_t size)
  {
    // find best fit block
    auto finder = [size](block_type const& lhs, block_type const& rhs) {
      return lhs.is_better_fit(size, rhs);
    };
    auto const iter = std::min_element(cbegin(), cend(), finder);

    if (iter != cend() && iter->fits(size)) {
      // Remove the block from the free_list and return it.
      block_type const found = *iter;
      erase(iter);
      return found;
    }

    return block_type{};  // not found
  }

#ifdef RMM_DEBUG_PRINT
  /**
   * @brief Print all blocks in the free_list.
   */
  void print() const
  {
    std::cout << size() << '\n';
    std::for_each(cbegin(), cend(), [](auto const iter) { iter.print(); });
  }
#endif
};  // coalescing_free_list

}  // namespace rmm::mr::detail
