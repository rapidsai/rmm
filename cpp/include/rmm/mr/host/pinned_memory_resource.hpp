/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
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
 *
 * @deprecated This class is deprecated in 25.12 and will be removed in 26.02.
 * Use `pinned_host_memory_resource` instead.
 */
class [[deprecated(
  "pinned_memory_resource is deprecated in 25.12 and will be removed in 26.02. "
  "Use pinned_host_memory_resource instead.")]] pinned_memory_resource final
  : public host_memory_resource {
 public:
  pinned_memory_resource()                              = default;
  ~pinned_memory_resource() override                    = default;
  pinned_memory_resource(pinned_memory_resource const&) = default;  ///< @default_copy_constructor
  pinned_memory_resource(pinned_memory_resource&&)      = default;  ///< @default_move_constructor
  pinned_memory_resource& operator=(pinned_memory_resource const&) =
    default;  ///< @default_copy_assignment{pinned_memory_resource}
  pinned_memory_resource& operator=(pinned_memory_resource&&) =
    default;  ///< @default_move_assignment{pinned_memory_resource}

#ifdef RMM_ENABLE_LEGACY_MR_INTERFACE
  /**
   * @brief Pretend to support the allocate_async interface, falling back to stream 0
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
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
   * @brief Pretend to support the allocate_async interface, falling back to stream 0
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
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
   * @brief Pretend to support the deallocate_async interface, falling back to stream 0
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   */
  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view) noexcept
  {
    do_deallocate(ptr, bytes);
  }

  // Explicitly inherit the allocate and deallocate functions from the host_memory_resource class.
  // Due to inheritance and name hiding rules, we need to declare these with "using" when we
  // override allocate and deallocate for CCCL 3.1.0+ compatibility.
  using host_memory_resource::allocate;
  using host_memory_resource::deallocate;
#endif  // RMM_ENABLE_LEGACY_MR_INTERFACE

  /**
   * @brief Pretend to support the allocate_async interface, falling back to stream 0
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param stream CUDA stream on which to perform the deallocation (ignored).
   * @param bytes The size of the allocation
   * @param alignment The expected alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(cuda_stream_view stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT)
  {
    return do_allocate(bytes, alignment);
  }

  /**
   * @brief Pretend to support the deallocate_async interface, falling back to stream 0
   *
   * @param stream CUDA stream on which to perform the deallocation (ignored).
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   */
  void deallocate(cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept
  {
    return do_deallocate(ptr, bytes, alignment);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_memory_resource` provides device accessible memory
   */
  friend void get_property(pinned_memory_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  /**
   * @brief Allocates pinned memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported, and to
   * `rmm::RMM_DEFAULT_HOST_ALIGNMENT` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes,
                    std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    RMM_EXPECTS(rmm::is_supported_alignment(alignment),
                "Allocation alignment is not a power of 2.",
                rmm::bad_alloc);

    return rmm::detail::aligned_host_allocate(bytes, alignment, [](std::size_t size) {
      void* ptr{nullptr};
      RMM_CUDA_TRY_ALLOC(cudaMallocHost(&ptr, size), size);
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
                     std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept override
  {
    if (nullptr == ptr) { return; }
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr)); });
  }
};

// static property checks
static_assert(rmm::detail::polyfill::async_resource_with<pinned_memory_resource,
                                                         cuda::mr::host_accessible,
                                                         cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
