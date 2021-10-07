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
    RMM_CUDA_TRY(cudaMemPoolCreate(&cuda_pool_handle_, &pool_props));

    auto const [free, total] = rmm::detail::available_device_memory();

    // Need an l-value to take address to pass to cudaMemPoolSetAttribute
    uint64_t threshold = release_threshold.value_or(total);
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(cuda_pool_handle_, cudaMemPoolAttrReleaseThreshold, &threshold));

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
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return cuda_pool_handle_; }
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
   * @return false
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

 private:
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  cudaMemPool_t cuda_pool_handle_{};
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
    if (bytes > 0) {
      RMM_CUDA_TRY(cudaMallocFromPoolAsync(&ptr, bytes, pool_handle(), stream.value()),
                   rmm::bad_alloc);
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
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view stream) override
  {
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
    if (ptr != nullptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeAsync(ptr, stream.value())); }
#else
    (void)ptr;
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
