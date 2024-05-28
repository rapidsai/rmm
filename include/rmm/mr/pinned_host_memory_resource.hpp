/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rmm/aligned.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/nvtx/ranges.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>

namespace rmm::mr {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Memory resource class for allocating pinned host memory.
 *
 * This class uses CUDA's `cudaHostAlloc` to allocate pinned host memory. It implements the
 * `cuda::mr::memory_resource` and `cuda::mr::device_memory_resource` concepts, and
 * the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.
 */
class pinned_host_memory_resource {
 public:
  // Disable clang-tidy complaining about the easily swappable size and alignment parameters
  // of allocate and deallocate
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * reason.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param alignment Alignment in bytes. Default alignment is used if unspecified.
   *
   * @return Pointer to the newly allocated memory.
   */
  static void* allocate(std::size_t bytes,
                        [[maybe_unused]] std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT)
  {
    RMM_FUNC_RANGE();

    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    return rmm::detail::aligned_host_allocate(bytes, alignment, [](std::size_t size) {
      void* ptr{nullptr};
      RMM_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
      return ptr;
    });
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr of size \p bytes bytes.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes Size of the allocation.
   * @param alignment Alignment in bytes. Default alignment is used if unspecified.
   */
  static void deallocate(void* ptr,
                         std::size_t bytes,
                         std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept
  {
    RMM_FUNC_RANGE();

    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr)); });
  }

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   *
   * @note Stream argument is ignored and behavior is identical to allocate.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * error.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param stream CUDA stream on which to perform the allocation (ignored).
   * @return Pointer to the newly allocated memory.
   */
  static void* allocate_async(std::size_t bytes, [[maybe_unused]] cuda::stream_ref stream)
  {
    RMM_FUNC_RANGE();

    return allocate(bytes);
  }

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes and alignment \p alignment.
   *
   * @note Stream argument is ignored and behavior is identical to allocate.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * error.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param alignment Alignment in bytes.
   * @param stream CUDA stream on which to perform the allocation (ignored).
   * @return Pointer to the newly allocated memory.
   */
  static void* allocate_async(std::size_t bytes,
                              std::size_t alignment,
                              [[maybe_unused]] cuda::stream_ref stream)
  {
    RMM_FUNC_RANGE();

    return allocate(bytes, alignment);
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr of size \p bytes bytes.
   *
   * @note Stream argument is ignored and behavior is identical to deallocate.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes Size of the allocation.
   * @param stream CUDA stream on which to perform the deallocation (ignored).
   */
  static void deallocate_async(void* ptr,
                               std::size_t bytes,
                               [[maybe_unused]] cuda::stream_ref stream) noexcept
  {
    RMM_FUNC_RANGE();

    return deallocate(ptr, bytes);
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr of size \p bytes bytes and alignment \p
   * alignment bytes.
   *
   * @note Stream argument is ignored and behavior is identical to deallocate.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes Size of the allocation.
   * @param alignment Alignment in bytes.
   * @param stream CUDA stream on which to perform the deallocation (ignored).
   */
  static void deallocate_async(void* ptr,
                               std::size_t bytes,
                               std::size_t alignment,
                               [[maybe_unused]] cuda::stream_ref stream) noexcept
  {
    RMM_FUNC_RANGE();

    return deallocate(ptr, bytes, alignment);
  }
  // NOLINTEND(bugprone-easily-swappable-parameters)

  /**
   * @briefreturn{true if the specified resource is the same type as this resource.}
   */
  bool operator==(const pinned_host_memory_resource&) const { return true; }

  /**
   * @briefreturn{true if the specified resource is not the same type as this resource, otherwise
   * false.}
   */
  bool operator!=(const pinned_host_memory_resource&) const { return false; }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  friend void get_property(pinned_host_memory_resource const&, cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  friend void get_property(pinned_host_memory_resource const&, cuda::mr::host_accessible) noexcept
  {
  }
};

static_assert(cuda::mr::async_resource_with<pinned_host_memory_resource,
                                            cuda::mr::device_accessible,
                                            cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace rmm::mr
