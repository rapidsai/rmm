/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/detail/thread_safe_resource_adaptor_impl.hpp>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

thread_safe_resource_adaptor_impl::thread_safe_resource_adaptor_impl(
  device_async_resource_ref upstream)
  : upstream_mr_{upstream}
{
}

device_async_resource_ref thread_safe_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

void* thread_safe_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t /*alignment*/)
{
  std::lock_guard<std::mutex> lock(mtx_);
  return upstream_mr_.allocate(stream, bytes);
}

void thread_safe_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t /*alignment*/) noexcept
{
  std::lock_guard<std::mutex> lock(mtx_);
  upstream_mr_.deallocate(stream, ptr, bytes);
}

void* thread_safe_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void thread_safe_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
