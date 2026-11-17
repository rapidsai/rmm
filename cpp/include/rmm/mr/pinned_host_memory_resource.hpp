/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>

RMM_NAMESPACE_BEGIN
namespace mr {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Callback invoked to initialize host memory before it is registered with CUDA.
 *
 * The callback receives the allocation pointer and size in bytes. It must return only after all
 * initialization is complete and must not deallocate or register the allocation. If the callback
 * throws, the host allocation is released and the exception is propagated.
 *
 * If the same callback is used by multiple threads, it is the caller's responsibility to ensure
 * that concurrent invocations are safe.
 */
using host_memory_initializer_t = std::function<void(void*, std::size_t)>;

/**
 * @brief Memory resource class for allocating pinned host memory.
 *
 * By default, this class uses CUDA's `cudaHostAlloc` to allocate pinned host memory. An optional
 * host memory initializer can instead be provided to initialize an ordinary host allocation before
 * it is pinned with `cudaHostRegister`. This lets callers control policies such as parallel page
 * touching and NUMA placement without the resource owning process-level policy.
 *
 * This class satisfies the `cuda::mr::resource` and `cuda::mr::synchronous_resource` concepts, and
 * the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.
 */
class RMM_EXPORT pinned_host_memory_resource final {
 public:
  pinned_host_memory_resource() = default;

  /**
   * @brief Constructs a pinned host memory resource using \p initializer.
   *
   * For each non-empty allocation, the resource allocates 256-byte-aligned host memory, invokes
   * `initializer` with the allocation pointer and requested size, and then registers the allocation
   * with `cudaHostRegister`. An empty initializer retains the default `cudaHostAlloc` behavior.
   *
   * @param initializer Callback invoked after host allocation and before CUDA registration
   */
  explicit pinned_host_memory_resource(host_memory_initializer_t initializer);

  ~pinned_host_memory_resource() = default;
  pinned_host_memory_resource(pinned_host_memory_resource const&) =
    default;  ///< @default_copy_constructor
  pinned_host_memory_resource(pinned_host_memory_resource&&) =
    default;  ///< @default_move_constructor
  pinned_host_memory_resource& operator=(pinned_host_memory_resource const&) =
    default;  ///< @default_copy_assignment{pinned_host_memory_resource}
  pinned_host_memory_resource& operator=(pinned_host_memory_resource&&) =
    default;  ///< @default_move_assignment{pinned_host_memory_resource}

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * reason.
   *
   * The stream argument is ignored.
   *
   * @param stream CUDA stream on which to perform the allocation (ignored).
   * @param bytes The size, in bytes, of the allocation.
   * @param alignment The alignment of the allocation
   *
   * @return Pointer to the newly allocated memory.
   */
  [[nodiscard]] void* allocate([[maybe_unused]] cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   *
   * The stream argument is ignored.
   *
   * @param stream This argument is ignored.
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes synchronously.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory.
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory pointed to by \p ptr synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(pinned_host_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Compare this resource to another.
   *
   * Resources are equivalent when both use `cudaHostAlloc`, or both use an initializer followed by
   * `cudaHostRegister`. The initializer itself does not participate in deallocation.
   *
   * @return true if allocations from either resource can be deallocated by the other
   */
  [[nodiscard]] bool operator==(pinned_host_memory_resource const&) const noexcept;

  /**
   * @copydoc operator==
   */
  [[nodiscard]] bool operator!=(pinned_host_memory_resource const&) const noexcept;

 private:
  host_memory_initializer_t initializer_{};
};

// static property checks
static_assert(cuda::mr::synchronous_resource<pinned_host_memory_resource>);
static_assert(cuda::mr::resource<pinned_host_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<pinned_host_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<pinned_host_memory_resource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<pinned_host_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<pinned_host_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
