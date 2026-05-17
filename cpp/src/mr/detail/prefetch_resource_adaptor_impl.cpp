/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/prefetch_resource_adaptor_impl.hpp>
#include <rmm/prefetch.hpp>

#include <cuda_runtime_api.h>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {
namespace {

bool is_prefetch_supported(
  cuda::mr::any_resource<cuda::mr::device_accessible>& upstream_mr) noexcept
{
  if (!rmm::detail::concurrent_managed_access::is_supported()) { return false; }

  auto constexpr size = rmm::CUDA_ALLOCATION_ALIGNMENT;
  rmm::cuda_stream stream{};
  void* ptr{};
  cudaError_t result{};
  try {
    ptr = upstream_mr.allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  } catch (...) {
    return false;
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  cudaMemLocation location{cudaMemLocationTypeDevice, rmm::get_current_cuda_device().value()};
  result = cudaMemPrefetchAsync(ptr, size, location, 0, stream.value());
#else
  result = cudaMemPrefetchAsync(ptr, size, rmm::get_current_cuda_device().value(), stream.value());
#endif
  upstream_mr.deallocate(stream, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const sync_result = cudaStreamSynchronize(stream.value());
  return result == cudaSuccess && sync_result == cudaSuccess;
}

}  // namespace

prefetch_resource_adaptor_impl::prefetch_resource_adaptor_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
  : upstream_mr_{std::move(upstream)}, prefetch_enabled_{is_prefetch_supported(upstream_mr_)}
{
}

device_async_resource_ref prefetch_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

void* prefetch_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                               std::size_t bytes,
                                               std::size_t alignment)
{
  void* ptr = upstream_mr_.allocate(stream, bytes, alignment);
  if (prefetch_enabled_) {
    try {
      rmm::prefetch(ptr, bytes, rmm::get_current_cuda_device(), cuda_stream_view{stream.get()});
    } catch (...) {
    }
  }
  return ptr;
}

void prefetch_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                void* ptr,
                                                std::size_t bytes,
                                                std::size_t alignment) noexcept
{
  upstream_mr_.deallocate(stream, ptr, bytes, alignment);
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
