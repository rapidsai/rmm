/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>

#include <cuda/memory_resource>

namespace RMM_NAMESPACE {
namespace mr {

cuda_async_managed_memory_resource::cuda_async_managed_memory_resource()
  : shared_base(cuda::mr::make_shared_resource<detail::cuda_async_managed_memory_resource_impl>())
{
}

cudaMemPool_t cuda_async_managed_memory_resource::pool_handle() const noexcept
{
  return get().pool_handle();
}

// Begin legacy device_memory_resource compatibility layer
void* cuda_async_managed_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void cuda_async_managed_memory_resource::do_deallocate(void* ptr,
                                                       std::size_t bytes,
                                                       cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool cuda_async_managed_memory_resource::do_is_equal(
  device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<cuda_async_managed_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
