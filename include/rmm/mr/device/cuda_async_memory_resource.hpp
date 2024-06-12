/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <rmm/detail/dynamic_load_runtime.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <optional>

#if CUDART_VERSION >= 11020  // 11.2 introduced cudaMallocAsync
#ifndef RMM_DISABLE_CUDA_MALLOC_ASYNC
#define RMM_CUDA_MALLOC_ASYNC_SUPPORT
#endif
#endif

namespace rmm::mr {
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
   * @note These values are exact copies from `cudaMemAllocationHandleType`. We need to
   * define our own enum here because the earliest CUDA runtime version that supports asynchronous
   * memory pools (CUDA 11.2) did not support these flags, so we need a placeholder that can be
   * used consistently in the constructor of `cuda_async_memory_resource` with all versions of
   * CUDA >= 11.2. See the `cudaMemAllocationHandleType` docs at
   * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
   */
  enum class allocation_handle_type {
    none                  = 0x0,  ///< Does not allow any export mechanism.
    posix_file_descriptor = 0x1,  ///< Allows a file descriptor to be used for exporting. Permitted
                                  ///< only on POSIX systems.
    win32     = 0x2,              ///< Allows a Win32 NT handle to be used for exporting. (HANDLE)
    win32_kmt = 0x4  ///< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
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
   * resource should support interprocess communication (IPC). Default is
   * `cudaMemHandleTypeNone` for no IPC support.
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size             = {},
                             std::optional<std::size_t> release_threshold             = {},
                             std::optional<allocation_handle_type> export_handle_type = {})
  {
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    // Check if cudaMallocAsync Memory pool supported
    RMM_EXPECTS(rmm::detail::async_alloc::is_supported(),
                "cudaMallocAsync not supported with this CUDA driver/runtime version");

    // Construct explicit pool
    cudaMemPoolProps pool_props{};
    pool_props.allocType   = cudaMemAllocationTypePinned;
    pool_props.handleTypes = static_cast<cudaMemAllocationHandleType>(
      export_handle_type.value_or(allocation_handle_type::none));
    RMM_EXPECTS(rmm::detail::async_alloc::is_export_handle_type_supported(pool_props.handleTypes),
                "Requested IPC memory handle type not supported");
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id   = rmm::get_current_cuda_device().value();
    cudaMemPool_t cuda_pool_handle{};
    RMM_CUDA_TRY(rmm::detail::async_alloc::cudaMemPoolCreate(&cuda_pool_handle, &pool_props));
    pool_ = cuda_async_view_memory_resource{cuda_pool_handle};

    // CUDA drivers before 11.5 have known incompatibilities with the async allocator.
    // We'll disable `cudaMemPoolReuseAllowOpportunistic` if cuda driver < 11.5.
    // See https://github.com/NVIDIA/spark-rapids/issues/4710.
    int driver_version{};
    RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
    constexpr auto min_async_version{11050};
    if (driver_version < min_async_version) {
      int disabled{0};
      RMM_CUDA_TRY(rmm::detail::async_alloc::cudaMemPoolSetAttribute(
        pool_handle(), cudaMemPoolReuseAllowOpportunistic, &disabled));
    }

    auto const [free, total] = rmm::available_device_memory();

    // Need an l-value to take address to pass to cudaMemPoolSetAttribute
    uint64_t threshold = release_threshold.value_or(total);
    RMM_CUDA_TRY(rmm::detail::async_alloc::cudaMemPoolSetAttribute(
      pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));

    // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
    // specified size
    auto const pool_size = initial_pool_size.value_or(free / 2);
    auto* ptr            = do_allocate(pool_size, cuda_stream_default);
    do_deallocate(ptr, pool_size, cuda_stream_default);
#else
    RMM_FAIL(
      "cudaMallocAsync not supported by the version of the CUDA Toolkit used for this build");
#endif
  }

#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return pool_.pool_handle(); }
#endif

  ~cuda_async_memory_resource() override
  {
#if defined(RMM_CUDA_MALLOC_ASYNC_SUPPORT)
    RMM_ASSERT_CUDA_SUCCESS(rmm::detail::async_alloc::cudaMemPoolDestroy(pool_handle()));
#endif
  }
  cuda_async_memory_resource(cuda_async_memory_resource const&)            = delete;
  cuda_async_memory_resource(cuda_async_memory_resource&&)                 = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&)      = delete;

 private:
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  cuda_async_view_memory_resource pool_{};
#endif

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
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    ptr = pool_.allocate(bytes, stream);
#else
    (void)bytes;
    (void)stream;
#endif
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
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    pool_.deallocate(ptr, bytes, stream);
#else
    (void)ptr;
    (void)bytes;
    (void)stream;
#endif
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
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    return (async_mr != nullptr) && (this->pool_handle() == async_mr->pool_handle());
#else
    return async_mr != nullptr;
#endif
  }
};

/** @} */  // end of group
}  // namespace rmm::mr
