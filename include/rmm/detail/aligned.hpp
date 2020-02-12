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

#include <cstdint>

namespace rmm {
namespace detail {

constexpr std::size_t RMM_DEFAULT_HOST_ALIGNMENT{alignof(std::max_align_t)};

constexpr bool is_pow2(std::size_t n) { return (0 == (n & (n - 1))); }

constexpr bool is_supported_alignment(std::size_t alignment) {
  return is_pow2(alignment);
}

/**
 * @brief Allocates sufficient memory to satisfy the requested size `bytes` with
 * alignment `alignment` using the unary callable `alloc` to allocate memory.
 *
 * Given a pointer `p` to an allocation of size `n` returned from the unary
 * callable `alloc`, the pointer `q` returned from `aligned_alloc` points to the
 * first location within the `n` bytes that satisfies `alignment`.
 *
 * In order to retrieve the original allocation pointer `p`, the difference
 * between `p` and `q` is stored in the `sizeof(std::size_t)` bytes at location
 * `q + bytes`.
 *
 * @param bytes The desired size of the allocation
 * @param alignment Desired alignment of allocation
 * @param alloc Unary callable given a size `n` will allocate at least `n` bytes
 * of host memory.
 * @return void* Pointer into allocation of at least `bytes` with desired
 * `alignment`.
 */
template <typename A>
void *aligned_allocate(std::size_t bytes, std::size_t alignment, A alloc) {
  // don't allocate anything if the user requested zero bytes
  if (0 == bytes) {
    return nullptr;
  }

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset
  void *p = alloc(bytes + alignment + sizeof(std::size_t));

  std::size_t ptr_int = reinterpret_cast<std::size_t>(p);

  // calculate the offset, i.e. how many bytes of correction were necessary
  // to get an aligned pointer
  std::size_t offset =
      (ptr_int % alignment) ? (alignment - ptr_int % alignment) : 0;
  // calculate the return pointer
  char *ptr = static_cast<char *>(p) + offset;
  // store the offset right after the actually returned value
  std::size_t *offset_store = reinterpret_cast<std::size_t *>(ptr + bytes);
  *offset_store = offset;
  return static_cast<void *>(ptr);
}

/**
 * @brief
 *
 * @tparam D
 * @param p
 * @param bytes
 * @param alignment
 * @param dealloc
 */
template <typename D>
void aligned_deallocate(void *p, std::size_t bytes, std::size_t alignment,
                        D dealloc) {
  (void)alignment;
  if (nullptr == p) {
    return;
  }
  char *ptr = static_cast<char *>(p);
  // calculate where the offset is stored
  std::size_t *offset = reinterpret_cast<std::size_t *>(ptr + bytes);
  // calculate the original pointer
  p = static_cast<void *>(ptr - *offset);
  dealloc(p);
}
}  // namespace detail
}  // namespace rmm