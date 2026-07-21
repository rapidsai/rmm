/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/cuda_async_managed_memory_resource_impl.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_EXPORT_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Memory resource that uses `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync`
 * with a managed memory pool for allocation/deallocation.
 */
class RMM_EXPORT cuda_async_managed_memory_resource final
  : public cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::cuda_async_managed_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Constructs a cuda_async_managed_memory_resource with the default managed memory pool for
   * the current device.
   *
   * The default managed memory pool is the pool that is created when the device is created.
   * Pool properties such as the release threshold are not modified.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocFromPoolAsync` with
   * managed memory pool
   */
  cuda_async_managed_memory_resource();

  /**
   * @brief Allocate memory using this resource.
   *
   * @param stream Stream on which to perform the allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory using this resource.
   *
   * @param stream Stream on which to perform deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Allocate memory synchronously using this resource.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory synchronously using this resource.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Compare two resources for equality.
   *
   * @param other The other resource to compare against
   * @return true if the resources compare equal, false otherwise
   */
  [[nodiscard]] bool operator==(cuda_async_managed_memory_resource const& other) const noexcept;
  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other resource to compare against
   * @return true if the resources do not compare equal, false otherwise
   */
  [[nodiscard]] bool operator!=(cuda_async_managed_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  ~cuda_async_managed_memory_resource();
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_managed_memory_resource}
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_managed_memory_resource}
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_async_managed_memory_resource>);
static_assert(cuda::mr::resource<cuda_async_managed_memory_resource>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_managed_memory_resource,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_managed_memory_resource,
                                                  cuda::mr::host_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_managed_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_managed_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_EXPORT_NAMESPACE
