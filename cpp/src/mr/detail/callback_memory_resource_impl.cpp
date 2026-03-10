/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/detail/callback_memory_resource_impl.hpp>

#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

callback_memory_resource_impl::callback_memory_resource_impl(
  std::function<void*(std::size_t, cuda_stream_view, void*)> allocate_callback,
  std::function<void(void*, std::size_t, cuda_stream_view, void*)> deallocate_callback,
  void* allocate_callback_arg,
  void* deallocate_callback_arg) noexcept
  : allocate_callback_(std::move(allocate_callback)),
    deallocate_callback_(std::move(deallocate_callback)),
    allocate_callback_arg_(allocate_callback_arg),
    deallocate_callback_arg_(deallocate_callback_arg)
{
}

void* callback_memory_resource_impl::allocate(cuda::stream_ref stream,
                                              std::size_t bytes,
                                              std::size_t /*alignment*/)
{
  return allocate_callback_(bytes, cuda_stream_view{stream.get()}, allocate_callback_arg_);
}

void callback_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                               void* ptr,
                                               std::size_t bytes,
                                               std::size_t /*alignment*/) noexcept
{
  deallocate_callback_(ptr, bytes, cuda_stream_view{stream.get()}, deallocate_callback_arg_);
}

void* callback_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void callback_memory_resource_impl::deallocate_sync(void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
