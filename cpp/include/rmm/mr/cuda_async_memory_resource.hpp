/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/cuda_async_memory_resource_impl.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>

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
class RMM_EXPORT cuda_async_memory_resource final
  : public cuda::mr::shared_resource<detail::cuda_async_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::cuda_async_memory_resource_impl>;

 public:
  /**
   * @brief Flags for specifying memory allocation handle types.
   *
   * @note These values are exact copies from `cudaMemAllocationHandleType`. We need a placeholder
   * that can be used consistently in the constructor of `cuda_async_memory_resource` with all
   * supported versions of CUDA. See the `cudaMemAllocationHandleType` docs at
   * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html and ensure the enum
   * values are kept in sync with the CUDA documentation.
   *
   * @note cudaMemHandleTypeFabric can be used instead of 0x8 once we require
   * CUDA 12.4+.
   */
  enum class allocation_handle_type : std::int32_t {
    none = cudaMemHandleTypeNone,  ///< Does not allow any export mechanism.
    posix_file_descriptor =
      cudaMemHandleTypePosixFileDescriptor,  ///< Allows a file descriptor to be used for exporting.
                                             ///< Permitted only on POSIX systems.
    win32 =
      cudaMemHandleTypeWin32,  ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
    win32_kmt = cudaMemHandleTypeWin32Kmt,  ///< Allows a Win32 KMT handle to be used for exporting.
                                            ///< (D3DKMT_HANDLE)
    fabric = 0x8  ///< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)
  };

  /**
   * @brief Flags for specifying memory pool usage.
   *
   * @note These values are exact copies from the runtime API. See the
   * `cudaMemPoolProps` docs at
   * https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemPoolProps.html
   * and ensure the enum values are kept in sync with the CUDA documentation.
   * `cudaMemPoolCreateUsageHwDecompress` is currently the only supported usage
   * flag, introduced in CUDA 12.8 and documented in
   * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
   */
  enum class mempool_usage : unsigned short {
    hw_decompress = 0x2,  ///< If set indicates that the memory can be used as a buffer for hardware
                          ///< accelerated decompression.
  };

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Constructs a cuda_async_memory_resource with the optionally specified initial pool size
   * and release threshold.
   *
   * If the pool size grows beyond the release threshold, unused memory held by the pool will be
   * released at the next synchronization event.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param initial_pool_size Optional initial size in bytes of the pool. If provided, the pool
   * will be primed by allocating and immediately deallocating this amount of memory on the
   * default CUDA stream.
   * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
   * provided, the release threshold is set to the total amount of memory on the current device.
   * @param export_handle_type Optional `cudaMemAllocationHandleType` that allocations from this
   * resource should support interprocess communication (IPC). Default is `cudaMemHandleTypeNone`
   * for no IPC support.
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size             = {},
                             std::optional<std::size_t> release_threshold             = {},
                             std::optional<allocation_handle_type> export_handle_type = {});

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  ~cuda_async_memory_resource() = default;
  cuda_async_memory_resource(cuda_async_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_memory_resource(cuda_async_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_memory_resource}
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_memory_resource}
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_async_memory_resource>);
static_assert(cuda::mr::resource<cuda_async_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<cuda_async_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_async_memory_resource, cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
