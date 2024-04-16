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
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>

#if CUDART_VERSION >= 11020  // 11.2 introduced cudaMallocAsync
#define RMM_CUDA_MALLOC_ASYNC_SUPPORT
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
class cuda_async_view_memory_resource final : public device_memory_resource {
 public:
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  /**
   * @brief Constructs a cuda_async_view_memory_resource which uses an existing CUDA memory pool.
   * The provided pool is not owned by cuda_async_view_memory_resource and must remain valid
   * during the lifetime of the memory resource.
   *
   * @throws rmm::runtime_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param valid_pool_handle Handle to a CUDA memory pool which will be used to
   * serve allocation requests.
   */
  cuda_async_view_memory_resource(cudaMemPool_t valid_pool_handle)
    : cuda_pool_handle_{[valid_pool_handle]() {
        RMM_EXPECTS(nullptr != valid_pool_handle, "Unexpected null pool handle.");
        return valid_pool_handle;
      }()}
  {
    // Check if cudaMallocAsync Memory pool supported
    auto const device = rmm::get_current_cuda_device();
    int cuda_pool_supported{};
    auto result =
      cudaDeviceGetAttribute(&cuda_pool_supported, cudaDevAttrMemoryPoolsSupported, device.value());
    RMM_EXPECTS(result == cudaSuccess && cuda_pool_supported,
                "cudaMallocAsync not supported with this CUDA driver/runtime version");
  }
#endif

#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return cuda_pool_handle_; }
#endif

  cuda_async_view_memory_resource() = default;
  cuda_async_view_memory_resource(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_view_memory_resource(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_view_memory_resource}
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_view_memory_resource}

 private:
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  cudaMemPool_t cuda_pool_handle_{};
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
    if (bytes > 0) {
      RMM_CUDA_TRY_ALLOC(rmm::detail::async_alloc::cudaMallocFromPoolAsync(
        &ptr, bytes, pool_handle(), stream.value()));
    }
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
  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     rmm::cuda_stream_view stream) override
  {
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    if (ptr != nullptr) {
      RMM_ASSERT_CUDA_SUCCESS(rmm::detail::async_alloc::cudaFreeAsync(ptr, stream.value()));
    }
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
    return dynamic_cast<cuda_async_view_memory_resource const*>(&other) != nullptr;
  }
};

/** @} */  // end of group
}  // namespace rmm::mr
