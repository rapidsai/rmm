/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_pinned_memory_resource_impl.hpp>

#include <cuda/memory_pool>
#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

cuda_async_pinned_memory_resource_impl::cuda_async_pinned_memory_resource_impl()
{
  RMM_EXPECTS(rmm::detail::runtime_async_pinned_alloc::is_supported(),
              "cuda_async_pinned_memory_resource is unsupported by this CUDA driver/runtime");
  pool_ = cuda_async_view_memory_resource{cuda::pinned_default_memory_pool().get()};
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
RMM_NAMESPACE_END
