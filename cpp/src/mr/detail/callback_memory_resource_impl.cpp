/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/callback_memory_resource_impl.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

callback_memory_resource_impl::callback_memory_resource_impl(
  std::function<void*(cuda_stream_view, std::size_t, std::size_t, void*)> allocate_callback,
  std::function<void(cuda_stream_view, void*, std::size_t, std::size_t, void*)> deallocate_callback,
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
                                              std::size_t alignment)
{
  RMM_EXPECTS(rmm::is_supported_alignment(alignment), "Allocation alignment is not a power of 2.");
  return allocate_callback_(
    cuda_stream_view{stream.get()}, bytes, alignment, allocate_callback_arg_);
}

void callback_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                               void* ptr,
                                               std::size_t bytes,
                                               std::size_t alignment) noexcept
{
  deallocate_callback_(
    cuda_stream_view{stream.get()}, ptr, bytes, alignment, deallocate_callback_arg_);
}

void* callback_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  auto* ptr         = allocate(stream, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return ptr;
}

void callback_memory_resource_impl::deallocate_sync(void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment) noexcept
{
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  deallocate(stream, ptr, bytes, alignment);
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
