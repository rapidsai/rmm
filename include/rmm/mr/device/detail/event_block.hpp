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

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
#include <cassert>
#include <cstddef>
#include <iostream>
#include <list>

#include <cuda_runtime_api.h>

namespace rmm {
namespace mr {
namespace detail {

struct event_block {
  event_block() = default;
  explicit event_block(char* ptr) : ptr{ptr}, size(0), is_head(true) {}
  event_block(char* ptr, size_t size, bool is_head) : ptr{ptr}, size{size}, is_head{is_head} {}
  event_block(char* ptr, size_t size, bool is_head, std::list<cudaEvent_t> const& events)
    : ptr{ptr}, size{size}, is_head{is_head}, events{events}
  {
  }

  /**
   * @brief Comparison operator to enable comparing blocks and storing in ordered containers.
   *
   * Orders by ptr address.

   * @param rhs
   * @return true if this block's ptr is < than `rhs` block pointer.
   * @return false if this block's ptr is >= than `rhs` block pointer.
   */
  inline bool operator<(event_block const& rhs) const noexcept { return ptr < rhs.ptr; };

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param b block to merge
   * @return block The merged block
   */
  inline event_block merge(event_block&& b) noexcept
  {
    assert(is_contiguous_before(b));
    size += b.size;
    events.splice(events.end(), std::move(b.events));
    b.ptr  = nullptr;
    b.size = 0;
    return *this;
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  inline bool is_contiguous_before(event_block const& b) const noexcept
  {
    return (this->ptr + this->size == b.ptr) and not(b.is_head);
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes
   */
  inline bool fits(size_t sz) const noexcept { return size >= sz; }

  /**
   * @brief Is this block a better fit for `sz` bytes than block `b`?
   *
   * @param sz The size in bytes to check for best fit.
   * @param b The other block to check for fit.
   * @return true If this block is a tighter fit for `sz` bytes than block `b`.
   * @return false If this block does not fit `sz` bytes or `b` is a tighter fit.
   */
  inline bool is_better_fit(size_t sz, event_block const& b) const noexcept
  {
    return fits(sz) && (size < b.size || b.size < sz);
  }

  /**
   * @brief Print this block. For debugging.
   */
  void print() const { std::cout << reinterpret_cast<void*>(ptr) << " " << size << "B\n"; }

  char* ptr;                      ///< Raw memory pointer
  size_t size;                    ///< Size in bytes
  bool is_head;                   ///< Indicates whether ptr was allocated from the heap
  std::list<cudaEvent_t> events;  ///< List of events to wait on before allocating from this block
};

};  // namespace detail
};  // namespace mr
};  // namespace rmm