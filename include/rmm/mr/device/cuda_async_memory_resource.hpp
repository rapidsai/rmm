/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_pool_wrapper.hpp>

#include <thrust/optional.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>

#if CUDART_VERSION >= 11020  // 11.2 introduced cudaMallocAsync
#define RMM_CUDA_MALLOC_ASYNC_SUPPORT
#endif

namespace rmm::mr {

/**
 * @brief `device_memory_resource` derived class that uses `cudaMallocAsync`/`cudaFreeAsync` for
 * allocation/deallocation.
 */
class cuda_async_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Constructs a cuda_async_memory_resource with the optionally specified initial pool size
   * and release threshold.
   *
   * If the pool size grows beyond the release threshold, unused memory held by the pool will be
   * released at the next synchronization event.
   *
   * @throws rmm::runtime_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param initial_pool_size Optional initial size in bytes of the pool. If no value is provided,
   * initial pool size is half of the available GPU memory.
   * @param release_threshold Optional release threshold size in bytes of the pool. If no value is
   * provided, the release threshold is set to the total amount of memory on the current device.
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  cuda_async_memory_resource(thrust::optional<std::size_t> initial_pool_size = {},
                             thrust::optional<std::size_t> release_threshold = {})
  {
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    // Check if cudaMallocAsync Memory pool supported
    auto const device = rmm::detail::current_device();
    int cuda_pool_supported{};
    auto result =
      cudaDeviceGetAttribute(&cuda_pool_supported, cudaDevAttrMemoryPoolsSupported, device.value());
    RMM_EXPECTS(result == cudaSuccess && cuda_pool_supported,
                "cudaMallocAsync not supported with this CUDA driver/runtime version");

    // Construct explicit pool
    cudaMemPoolProps pool_props{};
    pool_props.allocType     = cudaMemAllocationTypePinned;
    pool_props.handleTypes   = cudaMemHandleTypePosixFileDescriptor;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id   = device.value();

    cudaMemPool_t cuda_pool_handle{};
    RMM_CUDA_TRY(cudaMemPoolCreate(&cuda_pool_handle, &pool_props));

    auto const [free, total] = rmm::detail::available_device_memory();

    // Need an l-value to take address to pass to cudaMemPoolSetAttribute
    uint64_t threshold = release_threshold.value_or(total);
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(cuda_pool_handle, cudaMemPoolAttrReleaseThreshold, &threshold));

    pool_ = cuda_pool_wrapper{cuda_pool_handle};

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
   * @brief Constructs a cuda_async_memory_resource which uses an existing CUDA memory pool.
   * cuda_async_memory_resource takes ownership of this pool.
   *
   * @throws rmm::runtime_error if the CUDA version does not support `cudaMallocAsync`
   * @throws rmm::runtime_error if the pool is null
   * @throws rmm::runtime_error if the pool is the default memory pool of the current device
   *
   * @param valid_pool_handle Handle to a CUDA memory pool which will be used to
   * serve allocation requests. 
   */
  cuda_async_memory_resource(cudaMemPool_t valid_pool_handle)
    : pool_{[valid_pool_handle]() {
        RMM_EXPECTS(nullptr != valid_pool_handle, "Unexpected null pool handle.");        
        return valid_pool_handle;
      }()}
  {  
    // Check if cudaMallocAsync Memory pool supported
    auto const device = rmm::detail::current_device();
    int cuda_pool_supported{};
    auto result =
      cudaDeviceGetAttribute(&cuda_pool_supported, cudaDevAttrMemoryPoolsSupported, device.value());
    RMM_EXPECTS(result == cudaSuccess && cuda_pool_supported,
                "cudaMallocAsync not supported with this CUDA driver/runtime version");

    // Check if valid_pool_handle is not equal to to default memory pool
    cudaMemPool_t default_pool{};
    result = cudaDeviceGetDefaultMemPool(&default_pool, device.value());

    RMM_EXPECTS(result == cudaSuccess && default_pool != pool_handle(),
                "Cannot take ownership of the default memory pool");
  }
#endif

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
    RMM_ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool_handle()));
#endif
  }
  cuda_async_memory_resource(cuda_async_memory_resource const&) = delete;
  cuda_async_memory_resource(cuda_async_memory_resource&&)      = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) = delete;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&) = delete;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation. `cuda_memory_resource` does not support streams.
   *
   * @returns bool true
   */
  [[nodiscard]] bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return true
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

 private:

#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  cuda_pool_wrapper pool_{};
#endif

  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
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
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t size, rmm::cuda_stream_view stream) override
  {
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    if (ptr != nullptr) { pool_.deallocate(ptr, size, stream); }
#else
    (void)ptr;
    (void)size;
    (void)stream;
#endif
  }

  /**
   * @brief Compare this resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<cuda_async_memory_resource const*>(&other) != nullptr;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @return std::pair contaiing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

}  // namespace rmm::mr
