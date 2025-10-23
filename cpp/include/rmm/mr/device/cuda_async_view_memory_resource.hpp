/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

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
class cuda_async_view_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Constructs a cuda_async_view_memory_resource which uses an existing CUDA memory pool.
   * The provided pool is not owned by cuda_async_view_memory_resource and must remain valid
   * during the lifetime of the memory resource.
   *
   * @throws rmm::logic_error if the CUDA version does not support `cudaMallocAsync`
   *
   * @param pool_handle Handle to a CUDA memory pool which will be used to
   * serve allocation requests.
   */
  cuda_async_view_memory_resource(cudaMemPool_t pool_handle)
    : cuda_pool_handle_{[pool_handle]() {
        RMM_EXPECTS(nullptr != pool_handle, "Unexpected null pool handle.");
        return pool_handle;
      }()}
  {
    // Check if cudaMallocAsync Memory pool supported
    RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
                "cudaMallocAsync not supported with this CUDA driver/runtime version");
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return cuda_pool_handle_; }

  cuda_async_view_memory_resource()  = default;
  ~cuda_async_view_memory_resource() = default;
  cuda_async_view_memory_resource(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_constructor
  cuda_async_view_memory_resource(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_constructor
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_async_view_memory_resource}
  cuda_async_view_memory_resource& operator=(cuda_async_view_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_async_view_memory_resource}

 private:
  cudaMemPool_t cuda_pool_handle_{};

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
    if (bytes > 0) {
      RMM_CUDA_TRY_ALLOC(cudaMallocFromPoolAsync(&ptr, bytes, pool_handle(), stream.value()),
                         bytes);
    }
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
                     rmm::cuda_stream_view stream) noexcept override
  {
    if (ptr != nullptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeAsync(ptr, stream.value())); }
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
}  // namespace mr
}  // namespace RMM_NAMESPACE
