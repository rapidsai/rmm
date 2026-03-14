/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/runtime_capabilities.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Memory resource that uses `cudaMallocAsync`/`cudaFreeAsync` for
 * allocation/deallocation.
 */
class cuda_async_view_memory_resource final {
 public:
  /**
   * @brief Constructs a cuda_async_view_memory_resource which uses an existing CUDA memory pool.
   * The provided pool is not owned by cuda_async_view_memory_resource and must remain valid
   * during the lifetime of the memory resource.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param pool_handle Handle to a CUDA memory pool which will be used to
   * serve allocation requests.
   */
  cuda_async_view_memory_resource(cudaMemPool_t pool_handle)
    : cuda_pool_handle_{[pool_handle]() {
        RMM_EXPECTS(nullptr != pool_handle, "Unexpected null pool handle.");
        return pool_handle;
      }()}
  {
    // Check if cudaMallocAsync Memory pool supported
    RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
                "cudaMallocAsync not supported with this CUDA driver/runtime version");
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return cuda_pool_handle_; }

  cuda_async_view_memory_resource()  = default;
  ~cuda_async_view_memory_resource() = default;
  cuda_async_view_memory_resource(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_view_memory_resource(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_view_memory_resource}
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_view_memory_resource}

  // -- CCCL memory resource interface (hides device_memory_resource versions) --

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @param stream Stream on which to perform allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ptr{nullptr};
    if (bytes > 0) {
      RMM_CUDA_TRY_ALLOC(cudaMallocFromPoolAsync(&ptr, bytes, pool_handle(), stream.get()), bytes);
    }
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   *
   * @param stream Stream on which to perform deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  [[maybe_unused]] std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (ptr != nullptr) { RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFreeAsync(ptr, stream.get())); }
  }

  /**
   * @brief Allocates memory of size at least \p bytes synchronously.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = allocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(0)));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, ptr, bytes, alignment);
  }

  /**
   * @brief Compare this resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool operator==(cuda_async_view_memory_resource const& other) const noexcept
  {
    return pool_handle() == other.pool_handle();
  }

  /**
   * @copydoc operator==
   */
  [[nodiscard]] bool operator!=(cuda_async_view_memory_resource const& other) const noexcept
  {
    return !operator==(other);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `cuda_async_view_memory_resource` provides device accessible
   * memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_view_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cudaMemPool_t cuda_pool_handle_{};
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_async_view_memory_resource>);
static_assert(cuda::mr::resource<cuda_async_view_memory_resource>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_view_memory_resource,
                                                  cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_view_memory_resource, cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
