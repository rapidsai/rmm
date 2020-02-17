/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <list>
#include <set>
#include <algorithm>
#include <iostream>

namespace rmm {
namespace mr {
namespace detail {

struct block
{
  char* ptr;          ///< Raw memory pointer
  size_t size;        ///< Size in bytes
  bool is_head;       ///< Indicates whether ptr was allocated from the heap

  bool operator<(block const& rhs) const { return ptr < rhs.ptr; };

  void print() const {
    std::cout << reinterpret_cast<void*>(ptr) << " " << size << "B\n";
  }
};

// combine contiguous blocks
inline block merge_blocks(block const& a, block const& b)
{
  if (a.ptr + a.size != b.ptr || b.is_head)
    throw std::logic_error("Invalid block merge");

  return block{a.ptr, a.size + b.size};
}

template < typename list_type = std::list<block> >
struct free_list {

  using size_type = typename list_type::size_type;
  using iterator = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;

  iterator begin() noexcept              { return blocks.begin(); }
  const_iterator begin() const noexcept  { return begin(); }
  const_iterator cbegin() const noexcept { return begin(); }

  iterator end() noexcept                { return blocks.end(); }
  const_iterator end() const noexcept    { return end(); }
  const_iterator cend() const noexcept   { return end(); }

  size_type size() const noexcept        { return blocks.size(); }

  void insert(block const& b) {
    if (blocks.empty()) { 
      insert(blocks.end(), b);
      return;
    }

    auto next = std::find_if(blocks.begin(), blocks.end(),
                             [b](block const& i) { return i.ptr > b.ptr; });
    auto previous = (next == blocks.begin()) ? next : std::prev(next);

    bool merge_prev = !b.is_head && (previous->ptr + previous->size == b.ptr);
    bool merge_next = (next != blocks.end()) && !next->is_head && (b.ptr + b.size == next->ptr);

    if (merge_prev) {
      *previous = detail::merge_blocks(*previous, b);
    if (merge_next) {
      *previous = detail::merge_blocks(*previous, *next);
      erase(next);
    }
    } else if (merge_next) {
      *next = detail::merge_blocks(b, *next);
    } else {
      insert(next, b);
    }
  }

  template< class InputIt >
  void insert( InputIt first, InputIt last ) {
    for (auto iter = first; iter != last; ++iter) {
      insert(*iter);
    }
  }

  void erase(iterator const& iter) {
    blocks.erase(iter);
  }

  void clear() noexcept { blocks.clear(); }

  block best_fit(size_t size) {
    block dummy{nullptr, size, false};
    // find best fit block
    auto iter = std::min_element(blocks.begin(), blocks.end(),
    [size](block lhs, block rhs) {
      return (lhs.size >= size) && 
              ((lhs.size < rhs.size) || (rhs.size < size));
    });

    if (iter->size >= size)
    {
      block found = *iter;
      erase(iter);
      return found;
    }
    
    return dummy;
  }

  void print() const {
    std::cout << blocks.size() << "\n";
    for (block const& b : blocks) { b.print(); }
  }

protected:
  void insert(const_iterator pos, block const& b) {
    blocks.insert(pos, b);
  }

private:
  list_type blocks;
  //std::mutex blocks_mutex;
}; // free_list

} // namespace rmm::mr::detail
} // namespace rmm::mr
} // namespace rmm