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
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief `device_memory_resource` derived class that uses
 * `cudaMallocFromPoolAsync`/`cudaFreeFromPoolAsync` with a pinned memory pool
 * for allocation/deallocation.
 */
class cuda_async_pinned_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Constructs a cuda_async_pinned_memory_resource with a pinned memory pool for
   * the current device.
   *
   * On CUDA 12.6-12.x, creates a new pinned memory pool using cudaMemPoolCreate.
   * On CUDA 13.0+, uses the default pinned memory pool via cudaMemGetDefaultMemPool.
   *
   * Pool properties such as the release threshold are not modified.
   *
   * @throws rmm::logic_error if the CUDA build version is less than 12.6
   * @throws rmm::logic_error if the CUDA runtime version does not support pinned memory pools
   * (requires CUDA 12.6 or higher)
   */
  cuda_async_pinned_memory_resource()
  {
#if !defined(CUDA_VERSION) || CUDA_VERSION < RMM_MIN_ASYNC_PINNED_ALLOC_CUDA_VERSION
    RMM_FAIL(
      "cuda_async_pinned_memory_resource requires CUDA 12.6 or higher. "
      "This build was compiled with an older CUDA version.");
#else
    // Check if pinned memory pools are supported at runtime
    RMM_EXPECTS(rmm::detail::runtime_async_pinned_alloc::is_supported(),
                "cuda_async_pinned_memory_resource requires CUDA 12.6 or higher runtime");

    // Use host location for pinned memory pool (id is ignored for cudaMemLocationTypeHost)
    cudaMemLocation location{.type = cudaMemLocationTypeHost, .id = 0};

#if CUDA_VERSION >= 13000
    // CUDA 13.0+: Use the default pinned memory pool (no cleanup needed)
    RMM_CUDA_TRY(cudaMemGetDefaultMemPool(&pool_handle_, &location, cudaMemAllocationTypePinned));
#else
    // CUDA 12.6-12.x: Create a new pinned memory pool (needs cleanup)
    cudaMemPoolProps pool_props{};
    pool_props.allocType     = cudaMemAllocationTypePinned;
    pool_props.location.type = cudaMemLocationTypeHost;
    pool_props.location.id   = 0;
    RMM_CUDA_TRY(cudaMemPoolCreate(&pool_handle_, &pool_props));
    owns_pool_ = true;
#endif

    // Enable device access to the pinned memory pool
    cudaMemAccessDesc desc{};
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id   = rmm::get_current_cuda_device().value();
    desc.flags         = cudaMemAccessFlagsProtReadWrite;
    RMM_CUDA_TRY(cudaMemPoolSetAccess(pool_handle_, &desc, 1));

    pool_ = cuda_async_view_memory_resource{pool_handle_};
#endif
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return pool_.pool_handle(); }

  ~cuda_async_pinned_memory_resource() override
  {
#if defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_ASYNC_PINNED_ALLOC_CUDA_VERSION && \
  CUDA_VERSION < 13000
    if (owns_pool_ && pool_handle_ != nullptr) { cudaMemPoolDestroy(pool_handle_); }
#endif
  }
  cuda_async_pinned_memory_resource(cuda_async_pinned_memory_resource const&)            = delete;
  cuda_async_pinned_memory_resource(cuda_async_pinned_memory_resource&&)                 = delete;
  cuda_async_pinned_memory_resource& operator=(cuda_async_pinned_memory_resource const&) = delete;
  cuda_async_pinned_memory_resource& operator=(cuda_async_pinned_memory_resource&&)      = delete;

 private:
  cudaMemPool_t pool_handle_{};
  bool owns_pool_{false};
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
    auto const* async_mr = dynamic_cast<cuda_async_pinned_memory_resource const*>(&other);
    return (async_mr != nullptr) && (this->pool_handle() == async_mr->pool_handle());
  }

  friend auto get_property(cuda_async_pinned_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
    return cuda::mr::device_accessible{};
  }
  friend auto get_property(cuda_async_pinned_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
    return cuda::mr::host_accessible{};
  }
};

// static property checks
static_assert(rmm::detail::polyfill::resource<cuda_async_pinned_memory_resource>);
static_assert(rmm::detail::polyfill::async_resource<cuda_async_pinned_memory_resource>);
static_assert(rmm::detail::polyfill::resource_with<cuda_async_pinned_memory_resource,
                                                   cuda::mr::host_accessible,
                                                   cuda::mr::device_accessible>);
static_assert(rmm::detail::polyfill::async_resource_with<cuda_async_pinned_memory_resource,
                                                         cuda::mr::host_accessible,
                                                         cuda::mr::device_accessible>);
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
