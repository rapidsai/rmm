/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/runtime_async_alloc.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */

/**
 * @brief `device_memory_resource` derived class that uses `cudaMallocAsync`/`cudaFreeAsync` for
 * allocation/deallocation.
 */
class cuda_async_memory_resource final : public device_memory_resource {
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
   * @brief Constructs a cuda_async_memory_resource with the optionally specified initial pool size
   * and release threshold.
   *
   * If the pool size grows beyond the release threshold, unused memory held by the pool will be
   * released at the next synchronization event.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param initial_pool_size Optional initial size in bytes of the pool. If no value is provided,
   * initial pool size is half of the available GPU memory.
   * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
   * provided, the release threshold is set to the total amount of memory on the current device.
   * @param export_handle_type Optional `cudaMemAllocationHandleType` that allocations from this
   * resource should support interprocess communication (IPC). Default is `cudaMemHandleTypeNone`
   * for no IPC support.
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size             = {},
                             std::optional<std::size_t> release_threshold             = {},
                             std::optional<allocation_handle_type> export_handle_type = {})
  {
    // Check if cudaMallocAsync Memory pool supported
    RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
                "cudaMallocAsync not supported with this CUDA driver/runtime version");

    // Construct explicit pool
    cudaMemPoolProps pool_props{};
    pool_props.allocType   = cudaMemAllocationTypePinned;
    pool_props.handleTypes = static_cast<cudaMemAllocationHandleType>(
      export_handle_type.value_or(allocation_handle_type::none));

#if defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_DRIVER_VERSION
    // Enable hardware decompression if supported (requires CUDA 12.8 driver or higher)
    if (rmm::detail::runtime_async_alloc::is_hwdecompress_supported()) {
      pool_props.usage = static_cast<unsigned short>(mempool_usage::hw_decompress);
    }
#endif

    RMM_EXPECTS(
      rmm::detail::runtime_async_alloc::is_export_handle_type_supported(pool_props.handleTypes),
      "Requested IPC memory handle type not supported");
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id   = rmm::get_current_cuda_device().value();
    cudaMemPool_t cuda_pool_handle{};
    RMM_CUDA_TRY(cudaMemPoolCreate(&cuda_pool_handle, &pool_props));
    pool_ = cuda_async_view_memory_resource{cuda_pool_handle};

    auto const [free, total] = rmm::available_device_memory();

    // Need an l-value to take address to pass to cudaMemPoolSetAttribute
    uint64_t threshold = release_threshold.value_or(total);
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));

    // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
    // specified size
    auto const pool_size = initial_pool_size.value_or(free / 2);
    auto* ptr            = do_allocate(pool_size, cuda_stream_default);
    do_deallocate(ptr, pool_size, cuda_stream_default);
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return pool_.pool_handle(); }

  ~cuda_async_memory_resource() override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool_handle()));
  }
  cuda_async_memory_resource(cuda_async_memory_resource const&)            = delete;
  cuda_async_memory_resource(cuda_async_memory_resource&&)                 = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&)      = delete;

 private:
  cuda_async_view_memory_resource pool_{};

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    void* ptr{nullptr};
    ptr = pool_.allocate(bytes, stream);
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    pool_.deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare this resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    auto const* async_mr = dynamic_cast<cuda_async_memory_resource const*>(&other);
    return (async_mr != nullptr) && (this->pool_handle() == async_mr->pool_handle());
  }
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
