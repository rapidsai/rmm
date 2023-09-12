/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

#include <cstddef>
#include <utility>

namespace rmm::mr {

/**
 * @brief A `host_memory_resource` that uses `cudaMallocHost` to allocate
 * pinned/page-locked host memory.
 *
 * See https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */
class pinned_memory_resource final : public host_memory_resource {
 public:
  pinned_memory_resource()                              = default;
  ~pinned_memory_resource() override                    = default;
  pinned_memory_resource(pinned_memory_resource const&) = default;  ///< @default_copy_constructor
  pinned_memory_resource(pinned_memory_resource&&)      = default;  ///< @default_move_constructor
  pinned_memory_resource& operator=(pinned_memory_resource const&) =
    default;  ///< @default_copy_assignment{pinned_memory_resource}
  pinned_memory_resource& operator=(pinned_memory_resource&&) =
    default;  ///< @default_move_assignment{pinned_memory_resource}

 private:
  /**
   * @brief Allocates pinned memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported, and to
   * `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    // If the requested alignment isn't supported, use default
    alignment = (rmm::detail::is_supported_alignment(alignment))
                  ? alignment
                  : rmm::detail::RMM_DEFAULT_HOST_ALIGNMENT;

    return rmm::detail::aligned_allocate(bytes, alignment, [](std::size_t size) {
      void* ptr{nullptr};
      auto status = cudaMallocHost(&ptr, size);
      if (cudaSuccess != status) { throw std::bad_alloc{}; }
      return ptr;
    });
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * `ptr` must have been returned by a prior call to `allocate(bytes,alignment)` on a
   * `host_memory_resource` that compares equal to `*this`, and the storage it points to must not
   * yet have been deallocated, otherwise behavior is undefined.
   *
   * @throws Nothing.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   *               that was passed to the `allocate` call that returned `ptr`.
   * @param alignment Alignment of the allocation. This must be equal to the value of `alignment`
   *                  that was passed to the `allocate` call that returned `ptr`.
   */
  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     std::size_t alignment = alignof(std::max_align_t)) override
  {
    if (nullptr == ptr) { return; }
    rmm::detail::aligned_deallocate(
      ptr, bytes, alignment, [](void* ptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr)); });
  }
};
}  // namespace rmm::mr
