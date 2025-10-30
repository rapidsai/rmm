/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
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
 * @brief `device_memory_resource` derived class that uses
 * `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync` with a managed memory pool
 * for allocation/deallocation.
 */
class cuda_async_managed_memory_resource final : public device_memory_resource {
 public:
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
  cuda_async_managed_memory_resource()
  {
    // Check if managed memory pools are supported
    RMM_EXPECTS(rmm::detail::runtime_async_managed_alloc::is_supported(),
                "cuda_async_managed_memory_resource requires CUDA 13.0 or higher");

#if defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION
    cudaMemPool_t managed_pool_handle{};
    cudaMemLocation location{.type = cudaMemLocationTypeDevice,
                             .id   = rmm::get_current_cuda_device().value()};
    RMM_CUDA_TRY(
      cudaMemGetDefaultMemPool(&managed_pool_handle, &location, cudaMemAllocationTypeManaged));
    pool_ = cuda_async_view_memory_resource{managed_pool_handle};
#endif
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return pool_.pool_handle(); }

  ~cuda_async_managed_memory_resource() override {}
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource const&)            = delete;
  cuda_async_managed_memory_resource(cuda_async_managed_memory_resource&&)                 = delete;
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource const&) = delete;
  cuda_async_managed_memory_resource& operator=(cuda_async_managed_memory_resource&&)      = delete;

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
    return pool_.allocate(stream, bytes);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    pool_.deallocate(stream, ptr, bytes);
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
    auto const* async_mr = dynamic_cast<cuda_async_managed_memory_resource const*>(&other);
    return (async_mr != nullptr) && (this->pool_handle() == async_mr->pool_handle());
  }
};

// static property checks
static_assert(rmm::detail::polyfill::resource<cuda_async_managed_memory_resource>);
static_assert(rmm::detail::polyfill::async_resource<cuda_async_managed_memory_resource>);
static_assert(rmm::detail::polyfill::resource_with<cuda_async_managed_memory_resource,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::async_resource_with<cuda_async_managed_memory_resource,
                                                         cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
