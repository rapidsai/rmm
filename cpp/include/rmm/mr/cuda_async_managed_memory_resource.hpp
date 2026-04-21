/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/cuda_async_managed_memory_resource_impl.hpp>

#include <cuda/memory_resource>
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
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  ~cuda_async_managed_memory_resource() = default;
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
}  // namespace RMM_NAMESPACE
