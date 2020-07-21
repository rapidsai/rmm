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

#include <rmm/mr/device/detail/free_list.hpp>

#include <iostream>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief Checks whether a memory block is valid. Specialization for void* pointer blocks
 *
 * @param b The block to check for validity
 * @return true If `b` is valid (non-null)
 * @return false If `b` is not valid (null)
 */
template <>
inline bool is_valid<void*>(void* const& b)
{
  return b != nullptr;
}

struct fixed_size_free_list : free_list<void*> {
  fixed_size_free_list()  = default;
  ~fixed_size_free_list() = default;

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   *
   * @param b The block to insert.
   */
  virtual void insert(block_type const& b) override { push_back(b); }

  /**
   * @brief Splices blocks from range `[first, last)` onto the free_list.
   *
   * @param first The beginning of the range of blocks to insert
   * @param last The end of the range of blocks to insert.
   */
  virtual void insert(free_list&& other) override { splice(cend(), std::move(other)); }

  /**
   * @brief Returns the first block in the free list.
   *
   * @param size The size in bytes of the desired block (unused).
   * @return block A block large enough to store `size` bytes.
   */
  virtual block_type get_block(size_t size) override
  {
    if (is_empty())
      return block_type{};
    else {
      block_type b = *begin();
      pop_front();
      return b;
    }
  }

  /**
   * @brief Print all blocks in the free_list.
   */
  virtual void print() const override
  {
    std::cout << size() << '\n';
    for (const_iterator iter = begin(); iter != end(); ++iter) {
      std::cout << *iter;
    }
  }
};

}  // namespace detail
}  // namespace mr
}  // namespace rmm
