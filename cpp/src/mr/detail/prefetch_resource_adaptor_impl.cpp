/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/mr/detail/prefetch_resource_adaptor_impl.hpp>
#include <rmm/prefetch.hpp>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

prefetch_resource_adaptor_impl::prefetch_resource_adaptor_impl(device_async_resource_ref upstream)
  : upstream_mr_{upstream}
{
}

device_async_resource_ref prefetch_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

void* prefetch_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                               std::size_t bytes,
                                               std::size_t /*alignment*/)
{
  void* ptr = upstream_mr_.allocate(stream, bytes);
  rmm::prefetch(ptr, bytes, rmm::get_current_cuda_device(), cuda_stream_view{stream.get()});
  return ptr;
}

void prefetch_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                void* ptr,
                                                std::size_t bytes,
                                                std::size_t /*alignment*/) noexcept
{
  upstream_mr_.deallocate(stream, ptr, bytes);
}

void* prefetch_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void prefetch_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                     std::size_t bytes,
                                                     std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
