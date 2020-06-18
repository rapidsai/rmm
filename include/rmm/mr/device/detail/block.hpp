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

#include <cassert>
#include <iostream>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
struct block {
  block() = default;
  block(char* ptr, size_t size, bool is_head) : ptr{ptr}, size_bytes{size}, head{is_head} {}
  virtual ~block() = default;

  /**
   * @brief Returns the pointer to the memory represented by this block.
   *
   * @return the pointer to the memory represented by this block.
   */
  inline char* pointer() const { return ptr; }

  /**
   * @brief Returns the size of the memory represented by this block.
   *
   * @return the size in bytes of the memory represented by this block.
   */
  inline size_t size() const { return size_bytes; }

  /**
   * @brief Returns whether this block is the start of an allocation from an upstream allocator.
   *
   * A block `b` may not be coalesced with a preceding contiguous block `a` if `b.is_head == true`.
   *
   * @return true if this block is the start of an allocation from an upstream allocator.
   */
  inline bool is_head() const { return head; }

  /**
   * @brief Returns whether this block has a non-null pointer.
   *
   * @return true if this block has a non-null pointer, false otherwise.
   */
  inline bool is_valid() const { return pointer() != nullptr; }

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
   * @param b block to merge
   * @return block The merged block
   */
  inline block merge(block&& b) noexcept
  {
    assert(is_contiguous_before(b));
    size_bytes += b.size();
    return *this;
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  inline bool is_contiguous_before(block const& b) const noexcept
  {
    return (pointer() + size() == b.ptr) and not(b.is_head());
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes
   */
  inline bool fits(size_t sz) const noexcept { return size() >= sz; }

  /**
   * @brief Is this block a better fit for `sz` bytes than block `b`?
   *
   * @param sz The size in bytes to check for best fit.
   * @param b The other block to check for fit.
   * @return true If this block is a tighter fit for `sz` bytes than block `b`.
   * @return false If this block does not fit `sz` bytes or `b` is a tighter fit.
   */
  inline bool is_better_fit(size_t sz, block const& b) const noexcept
  {
    return fits(sz) && (size() < b.size() || b.size() < sz);
  }

  /**
   * @brief Print this block. For debugging.
   */
  void print() const { std::cout << reinterpret_cast<void*>(pointer()) << " " << size() << "B\n"; }

 protected:
  char* ptr{};          ///< Raw memory pointer
  size_t size_bytes{};  ///< Size in bytes
  bool head{};          ///< Indicates whether ptr was allocated from the heap
};

/**
 * @brief Comparator for block types based on pointer address.
 *
 * This comparator allows searching associative containers of blocks by pointer rather than
 * having to search by the contained type. Saves potentially error-prone temporary construction of
 * a block when you just want to search by pointer.
 */
template <typename block_type>
struct compare_blocks {
  // is_transparent (C++14 feature) allows different search key types in set<block_type>::find()
  using is_transparent = void;

  bool operator()(block_type const& lhs, block_type const& rhs) const { return lhs < rhs; }
  bool operator()(char const* ptr, block_type const& rhs) const { return ptr < rhs.pointer(); }
  bool operator()(block_type const& lhs, char const* ptr) const { return lhs.pointer() < ptr; };
};

}  // namespace detail
}  // namespace mr
}  // namespace rmm