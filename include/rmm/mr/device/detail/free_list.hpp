/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <rmm/detail/export.hpp>

#include <algorithm>
#ifdef RMM_DEBUG_PRINT
#include <iostream>
#endif
#include <list>

namespace RMM_NAMESPACE {
namespace mr::detail {

struct block_base {
  void* ptr{};  ///< Raw memory pointer

  block_base() = default;
  block_base(void* ptr) : ptr{ptr} {};

  /// Returns the raw pointer for this block
  [[nodiscard]] inline void* pointer() const { return ptr; }
  /// Returns true if this block is valid (non-null), false otherwise
  [[nodiscard]] inline bool is_valid() const { return pointer() != nullptr; }

#ifdef RMM_DEBUG_PRINT
  /// Prints the block to stdout
  inline void print() const { std::cout << pointer(); }
#endif
};

#ifdef RMM_DEBUG_PRINT
/// Print block_base on an ostream
inline std::ostream& operator<<(std::ostream& out, const block_base& block)
{
  out << block.pointer();
  return out;
}
#endif

/**
 * @brief Base class defining an interface for a list of free memory blocks.
 *
 * Derived classes typically provide additional methods such as the following (see
 * fixed_size_free_list.hpp and coalescing_free_list.hpp). However this is not a required interface.
 *
 *  - `void insert(block_type const& b)  // insert a block into the free list`
 *  - `void insert(free_list&& other)    // insert / merge another free list`
 *  - `block_type get_block(std::size_t size) // get a block of at least size bytes
 *  - `void print()                      // print the block`
 *
 * @tparam list_type the type of the internal list data structure.
 */
template <typename BlockType, typename ListType = std::list<BlockType>>
class free_list {
 public:
  free_list()          = default;
  virtual ~free_list() = default;

  free_list(free_list const&)            = delete;
  free_list& operator=(free_list const&) = delete;
  free_list(free_list&&)                 = delete;
  free_list& operator=(free_list&&)      = delete;

  using block_type     = BlockType;
  using list_type      = ListType;
  using size_type      = typename list_type::size_type;
  using iterator       = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;

  /// beginning of the free list
  [[nodiscard]] iterator begin() noexcept { return blocks.begin(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator begin() const noexcept { return blocks.begin(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator cbegin() const noexcept { return blocks.cbegin(); }

  /// end of the free list
  [[nodiscard]] iterator end() noexcept { return blocks.end(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator end() const noexcept { return blocks.end(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator cend() const noexcept { return blocks.cend(); }

  /**
   * @brief The size of the free list in blocks.
   *
   * @return size_type The number of blocks in the free list.
   */
  [[nodiscard]] size_type size() const noexcept { return blocks.size(); }

  /**
   * @brief checks whether the free_list is empty.
   *
   * @return true If there are blocks in the free_list.
   * @return false If there are no blocks in the free_list.
   */
  [[nodiscard]] bool is_empty() const noexcept { return blocks.empty(); }

  /**
   * @brief Removes the block indicated by `iter` from the free list.
   *
   * @param iter An iterator referring to the block to erase.
   */
  void erase(const_iterator iter) { blocks.erase(iter); }

  /**
   * @brief Erase all blocks from the free_list.
   *
   */
  void clear() noexcept { blocks.clear(); }

#ifdef RMM_DEBUG_PRINT
  /**
   * @brief Print all blocks in the free_list.
   */
  void print() const
  {
    std::cout << size() << std::endl;
    for (auto const& block : blocks) {
      std::cout << block << std::endl;
    }
  }
#endif

 protected:
  /**
   * @brief Insert a block in the free list before the specified position
   *
   * @param pos iterator before which the block will be inserted. pos may be the end() iterator.
   * @param block The block to insert.
   */
  void insert(const_iterator pos, block_type const& block) { blocks.insert(pos, block); }

  /**
   * @brief Inserts a list of blocks in the free list before the specified position
   *
   * @param pos iterator before which the block will be inserted. pos may be the end() iterator.
   * @param other The free list to insert.
   */
  void splice(const_iterator pos, free_list&& other)
  {
    return blocks.splice(pos, std::move(other.blocks));
  }

  /**
   * @brief Appends the given block to the end of the free list.
   *
   * @param block The block to append.
   */
  void push_back(const block_type& block) { blocks.push_back(block); }

  /**
   * @brief Appends the given block to the end of the free list. `b` is moved to the new element.
   *
   * @param block The block to append.
   */
  void push_back(block_type&& block) { blocks.push_back(std::move(block)); }

  /**
   * @brief Removes the first element of the free list. If there are no elements in the free list,
   * the behavior is undefined.
   *
   * References and iterators to the erased element are invalidated.
   */
  void pop_front() { blocks.pop_front(); }

 private:
  list_type blocks;  // The internal container of blocks
};

}  // namespace mr::detail
}  // namespace RMM_NAMESPACE
