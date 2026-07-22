/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {

cuda_async_view_memory_resource::cuda_async_view_memory_resource(cudaMemPool_t pool_handle)
  : cuda_pool_handle_{[pool_handle]() {
      RMM_EXPECTS(nullptr != pool_handle, "Unexpected null pool handle.");
      return pool_handle;
    }()}
{
  // Check if cudaMallocAsync Memory pool supported
  RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
              "cudaMallocAsync not supported with this CUDA driver/runtime version");
}

cudaMemPool_t cuda_async_view_memory_resource::pool_handle() const noexcept
{
  return cuda_pool_handle_;
}

void* cuda_async_view_memory_resource::allocate(cuda::stream_ref stream,
                                                std::size_t bytes,
                                                std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  RMM_EXPECTS(rmm::is_supported_base_resource_alignment(alignment),
              "Requested alignment is larger than this memory resource supports.",
              rmm::bad_alloc);
  void* ptr{nullptr};
  RMM_CUDA_TRY_ALLOC(cudaMallocFromPoolAsync(&ptr, bytes, pool_handle(), stream.get()), bytes);
  return ptr;
}

void cuda_async_view_memory_resource::deallocate(cuda::stream_ref stream,
                                                 void* ptr,
                                                 [[maybe_unused]] std::size_t bytes,
                                                 [[maybe_unused]] std::size_t alignment) noexcept
{
  if (ptr != nullptr) { RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFreeAsync(ptr, stream.get())); }
}

void* cuda_async_view_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void cuda_async_view_memory_resource::deallocate_sync(void* ptr,
                                                      std::size_t bytes,
                                                      std::size_t alignment) noexcept
{
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  deallocate(stream, ptr, bytes, alignment);
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
}

bool cuda_async_view_memory_resource::operator==(
  cuda_async_view_memory_resource const& other) const noexcept
{
  return pool_handle() == other.pool_handle();
}

bool cuda_async_view_memory_resource::operator!=(
  cuda_async_view_memory_resource const& other) const noexcept
{
  return !operator==(other);
}

}  // namespace mr
RMM_NAMESPACE_END
