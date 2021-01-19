/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include "rmm/cuda_stream_view.hpp"

#include <cuda_runtime_api.h>

namespace rmm {
namespace mr {
/**
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 */
class cuda_async_memory_resource final : public device_memory_resource {
 public:
  cuda_async_memory_resource() = default;

  ~cuda_async_memory_resource()
  {
    cudaDeviceSynchronize();
    int device{0};
    RMM_ASSERT_CUDA_SUCCESS(cudaGetDevice(&device));
    cudaMemPool_t pool;
    RMM_ASSERT_CUDA_SUCCESS(cudaDeviceGetDefaultMemPool(&pool, device));
    RMM_ASSERT_CUDA_SUCCESS(cudaMemPoolTrimTo(pool, 0));
  }

  cuda_async_memory_resource(cuda_async_memory_resource const&) = default;
  cuda_async_memory_resource(cuda_async_memory_resource&&)      = default;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource const&) = default;
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&) = default;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation. `cuda_memory_resource` does not support streams.
   *
   * @returns bool true
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return true
   */
  bool supports_get_mem_info() const noexcept override { return true; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @note Stream argument is ignored
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    void* p{nullptr};
    if (bytes > 0) { RMM_CUDA_TRY(cudaMallocAsync(&p, bytes, stream.value()), rmm::bad_alloc); }
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t, rmm::cuda_stream_view stream) override
  {
    if (p != nullptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeAsync(p, stream.value())); }
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two cuda_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const& other) const noexcept override
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
  std::pair<size_t, size_t> do_get_mem_info(rmm::cuda_stream_view) const override
  {
    std::size_t free_size;
    std::size_t total_size;
    RMM_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    return std::make_pair(free_size, total_size);
  }
};
}  // namespace mr
}  // namespace rmm
