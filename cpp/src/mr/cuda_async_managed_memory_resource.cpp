/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

cuda_async_managed_memory_resource::cuda_async_managed_memory_resource()
  : shared_base(cuda::mr::make_shared_resource<detail::cuda_async_managed_memory_resource_impl>())
{
}

cuda_async_managed_memory_resource::~cuda_async_managed_memory_resource() = default;

void* cuda_async_managed_memory_resource::allocate(cuda::stream_ref stream,
                                                   std::size_t bytes,
                                                   std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void cuda_async_managed_memory_resource::deallocate(cuda::stream_ref stream,
                                                    void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* cuda_async_managed_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void cuda_async_managed_memory_resource::deallocate_sync(void* ptr,
                                                         std::size_t bytes,
                                                         std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool cuda_async_managed_memory_resource::operator==(
  cuda_async_managed_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

cudaMemPool_t cuda_async_managed_memory_resource::pool_handle() const noexcept
{
  return get().pool_handle();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
