/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_managed_memory_resource_impl.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

cuda_async_managed_memory_resource_impl::cuda_async_managed_memory_resource_impl()
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

cudaMemPool_t cuda_async_managed_memory_resource_impl::pool_handle() const noexcept
{
  return pool_.pool_handle();
}

void* cuda_async_managed_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                        std::size_t bytes,
                                                        std::size_t /*alignment*/)
{
  return pool_.allocate(stream, bytes);
}

void cuda_async_managed_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                         void* ptr,
                                                         std::size_t bytes,
                                                         std::size_t /*alignment*/) noexcept
{
  pool_.deallocate(stream, ptr, bytes);
}

void* cuda_async_managed_memory_resource_impl::allocate_sync(std::size_t bytes,
                                                             std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void cuda_async_managed_memory_resource_impl::deallocate_sync(void* ptr,
                                                              std::size_t bytes,
                                                              std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
