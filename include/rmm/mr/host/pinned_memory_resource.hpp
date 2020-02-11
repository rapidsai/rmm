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

#include "host_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <utility>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief A `host_memory_resource` that uses `cudaMallocHost` to allocate
 * pinned/page-locked host memory.
 *
 * See https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 *---------------------------------------------------------------------------**/
class pinned_memory_resource final : public host_memory_resource {
 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates pinned memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void *do_allocate(std::size_t bytes, std::size_t alignment =
                                           alignof(std::max_align_t)) override {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) {
      return nullptr;
    }
    // allocate memory for bytes, plus potential alignment correction,
    // plus store of the correction offset
    void *p{nullptr};
    auto status = cudaMallocHost(&p, (bytes + alignment + sizeof(std::size_t)));
    if (cudaSuccess != status) {
      throw std::bad_alloc{};
    }

    std::size_t ptr_int = reinterpret_cast<std::size_t>(p);
    // calculate the offset, i.e. how many bytes of correction was necessary
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

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,alignment)`
   * on a `host_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void do_deallocate(
      void *p, std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) override {
    (void)alignment;
    if (nullptr == p) {
      return;
    }
    char *ptr = static_cast<char *>(p);
    // calculate where the offset is stored
    std::size_t *offset = reinterpret_cast<std::size_t *>(ptr + bytes);
    // calculate the original pointer
    p = static_cast<void *>(ptr - *offset);
    auto status = cudaFreeHost(p);
    assert(status == cudaSuccess);
  }
};
}  // namespace mr
}  // namespace rmm
