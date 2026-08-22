/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_pinned_memory_resource_impl.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {
namespace {

void enable_access_from_all_devices(cudaMemPool_t pool_handle)
{
  int device_count{};
  RMM_CUDA_TRY(cudaGetDeviceCount(&device_count));
  for (int device = 0; device < device_count; ++device) {
    cudaMemAccessDesc access{};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id   = device;
    access.flags         = cudaMemAccessFlagsProtReadWrite;
    RMM_CUDA_TRY(cudaMemPoolSetAccess(pool_handle, &access, 1));
  }
}

cudaMemPool_t create_process_wide_pinned_pool()
{
  cudaMemPool_t pool_handle{};

#if CUDART_VERSION >= 13000
  cudaMemLocation location{.type = cudaMemLocationTypeHost, .id = 0};
  RMM_CUDA_TRY(cudaMemGetDefaultMemPool(&pool_handle, &location, cudaMemAllocationTypePinned));
#else
  cudaMemPoolProps properties{};
  properties.allocType     = cudaMemAllocationTypePinned;
  properties.handleTypes   = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeHostNuma;
  properties.location.id   = 0;
  RMM_CUDA_TRY(cudaMemPoolCreate(&pool_handle, &properties));
#endif

  try {
    enable_access_from_all_devices(pool_handle);
  } catch (...) {
#if CUDART_VERSION < 13000
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaMemPoolDestroy(pool_handle));
#endif
    throw;
  }

  // CUDA 12.x does not provide a default pinned host pool. Keep the custom pool alive for the
  // process lifetime so every resource instance remains equivalent and can deallocate allocations
  // made by any other instance. CUDA owns the default pool returned on CUDA 13 and later.
  return pool_handle;
}

cudaMemPool_t get_process_wide_pinned_pool()
{
  static cudaMemPool_t const pool_handle = create_process_wide_pinned_pool();
  return pool_handle;
}

}  // namespace

cuda_async_pinned_memory_resource_impl::cuda_async_pinned_memory_resource_impl()
{
  RMM_EXPECTS(rmm::detail::runtime_async_pinned_alloc::is_supported(),
              "cuda_async_pinned_memory_resource is unsupported by this CUDA driver/runtime");
  pool_ = cuda_async_view_memory_resource{get_process_wide_pinned_pool()};
}

cuda_async_pinned_memory_resource_impl::cuda_async_pinned_memory_resource_impl(construction_tag)
  : cuda_async_pinned_memory_resource_impl()
{
}

cudaMemPool_t cuda_async_pinned_memory_resource_impl::pool_handle() const noexcept
{
  return pool_.pool_handle();
}

void* cuda_async_pinned_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                       std::size_t bytes,
                                                       std::size_t alignment)
{
  return pool_.allocate(stream, bytes, alignment);
}

void cuda_async_pinned_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                        void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t /*alignment*/) noexcept
{
  pool_.deallocate(stream, ptr, bytes);
}

void* cuda_async_pinned_memory_resource_impl::allocate_sync(std::size_t bytes,
                                                            std::size_t alignment)
{
  return pool_.allocate_sync(bytes, alignment);
}

void cuda_async_pinned_memory_resource_impl::deallocate_sync(void* ptr,
                                                             std::size_t bytes,
                                                             std::size_t alignment) noexcept
{
  pool_.deallocate_sync(ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
