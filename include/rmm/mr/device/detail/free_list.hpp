/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithm>
#include <list>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief Checks whether a memory block is valid.
 *
 * This function template should be specialized for any type of block used by a concrete
 * implementation of `free_list`.
 *
 * @param b The block to check for validity
 * @return true If `b` is valid
 * @return false If `b` is not valid
 */
template <typename BlockType>
inline bool is_valid(BlockType const& b)
{
  return false;
}

/**
 * @brief Abstract base class defining an interface for a list of free memory blocks.
 *
 * Derived classes must implement:
 *  - virtual void insert(block_type const& b)
 *  -
 * @tparam list_type the type of the internal list data structure.
 */
template <typename BlockType, typename ListType = std::list<BlockType>>
struct free_list {
  free_list()          = default;
  virtual ~free_list() = default;

  using block_type     = BlockType;
  using list_type      = ListType;
  using size_type      = typename list_type::size_type;
  using iterator       = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;

  iterator begin() noexcept { return blocks.begin(); }                /// beginning of the free list
  const_iterator begin() const noexcept { return begin(); }           /// beginning of the free list
  const_iterator cbegin() const noexcept { return blocks.cbegin(); }  /// beginning of the free list

  iterator end() noexcept { return blocks.end(); }                /// end of the free list
  const_iterator end() const noexcept { return end(); }           /// end of the free list
  const_iterator cend() const noexcept { return blocks.cend(); }  /// end of the free list

  /**
   * @brief The size of the free list in blocks.
   *
   * @return size_type The number of blocks in the free list.
   */
  size_type size() const noexcept { return blocks.size(); }

  /**
   * @brief checks whether the free_list is empty.
   *
   * @return true If there are blocks in the free_list.
   * @return false If there are no blocks in the free_list.
   */
  bool is_empty() const noexcept { return blocks.empty(); }

  /**
   * @brief Inserts a block into the `free_list`.
   *
   * @param b The block to insert.
   */
  virtual void insert(block_type const& b) = 0;

  /**
   * @brief Moves blocks from other into the free_list.
   *
   * @param first The beginning of the range of blocks to insert
   * @param last The end of the range of blocks to insert.
   */
  virtual void insert(free_list&& other) = 0;

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

  /**
   * @brief Returns a block from `free_list` large enough to fit `size` bytes.
   *
   * @param size The size in bytes of the desired block.
   * @return block A block large enough to store `size` bytes.
   */
  virtual block_type get_block(size_t size) = 0;

  /**
   * @brief Print all blocks in the free_list.
   */
  virtual void print() const = 0;

 protected:
  /**
   * @brief Insert a block in the free list before the specified position
   *
   * @param pos iterator before which the block will be inserted. pos may be the end() iterator.
   * @param b The block to insert.
   */
  void insert(const_iterator pos, block_type const& b) { blocks.insert(pos, b); }

  /**
   * @brief Inserts a list of blocks in the free list before the specified position
   *
   * @param pos iterator before which the block will be inserted. pos may be the end() iterator.
   * @param b The block to insert.
   */
  void splice(const_iterator pos, free_list&& other) { return blocks.splice(pos, other.blocks); }

  /**
   * @brief Appends the given block to the end of the free list.
   *
   * @param b The block to append.
   */
  void push_back(const block_type& b) { blocks.push_back(b); }

  /**
   * @brief Appends the given block to the end of the free list. `b` is moved to the new element.
   *
   * @param b The block to append.
   */
  void push_back(block_type&& b) { blocks.push_back(b); }

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

}  // namespace detail
}  // namespace mr
}  // namespace rmm
