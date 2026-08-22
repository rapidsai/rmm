/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/cuda_async_pinned_memory_resource_impl.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

RMM_NAMESPACE_BEGIN
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Stream-ordered memory resource for allocating pinned host memory.
 *
 * This resource uses `cudaMallocFromPoolAsync` and `cudaFreeAsync`. Allocations are ordered on the
 * supplied stream and must not be accessed from the host until the allocation reaches the head of
 * that stream. Deallocations are ordered after preceding work on the supplied stream.
 *
 * All instances use CCCL's process-wide default pinned memory pool. With CUDA 12.x, this pool is on
 * NUMA node 0. With CUDA 13.0 and later, this is CUDA's default pinned host memory pool.
 * Allocations are accessible from all visible CUDA devices. CCCL configures the pool to retain
 * unused memory by setting its release threshold to the maximum value.
 */
class RMM_EXPORT cuda_async_pinned_memory_resource final
  : public cuda::mr::shared_resource<detail::cuda_async_pinned_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::cuda_async_pinned_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_pinned_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_pinned_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Constructs a resource that uses the process-wide pinned host memory pool.
   *
   * @throws rmm::logic_error if stream-ordered pinned host allocation is unsupported
   * @throws cuda::cuda_error if the pinned host memory pool cannot be initialized
   */
  cuda_async_pinned_memory_resource();

  /**
   * @brief Returns the underlying native CUDA memory pool handle.
   *
   * @return Handle to the underlying CUDA memory pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  ~cuda_async_pinned_memory_resource() = default;
  cuda_async_pinned_memory_resource(cuda_async_pinned_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_pinned_memory_resource(cuda_async_pinned_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_pinned_memory_resource& operator=(cuda_async_pinned_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_pinned_memory_resource}
  cuda_async_pinned_memory_resource& operator=(cuda_async_pinned_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_pinned_memory_resource}
};

static_assert(cuda::mr::synchronous_resource<cuda_async_pinned_memory_resource>);
static_assert(cuda::mr::resource<cuda_async_pinned_memory_resource>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_pinned_memory_resource,
                                                  cuda::mr::device_accessible>);
static_assert(cuda::mr::synchronous_resource_with<cuda_async_pinned_memory_resource,
                                                  cuda::mr::host_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_pinned_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::resource_with<cuda_async_pinned_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
