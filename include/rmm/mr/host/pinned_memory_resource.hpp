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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

#include <cstddef>
#include <utility>

namespace rmm::mr {
/**
 * @addtogroup host_memory_resources
 * @{
 * @file
 */

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

  /**
   * @brief Query whether the pinned_memory_resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool false.
   */
  [[nodiscard]] bool supports_streams() const noexcept { return false; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool false.
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept { return false; }

  /**
   * @brief Queries the amount of free and total memory for the resource.
   *
   * @param stream the stream whose memory manager we want to retrieve
   *
   * @returns a pair containing the free memory in bytes in .first and total amount of memory in
   * .second
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> get_mem_info(cuda_stream_view stream) const
  {
    return std::make_pair(0, 0);
  }

  /**
   * @brief Pretent to support the allocate_async interface, falling back to stream 0
   *
   * @throws `rmm::bad_alloc` When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param alignment The expected alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view)
  {
    return do_allocate(bytes, alignment);
  }

  /**
   * @brief Pretent to support the allocate_async interface, falling back to stream 0
   *
   * @throws `rmm::bad_alloc` When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_async(std::size_t bytes, cuda_stream_view)
  {
    return do_allocate(bytes);
  }

  /**
   * @brief Pretent to support the deallocate_async interface, falling back to stream 0
   *
   * @throws Nothing.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   */
  void deallocate_async(void* ptr, std::size_t bytes, std::size_t alignment, cuda_stream_view)
  {
    do_deallocate(ptr, rmm::detail::align_up(bytes, alignment));
  }

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
static_assert(cuda::mr::async_resource<pinned_memory_resource>);
/** @} */  // end of group
}  // namespace rmm::mr
