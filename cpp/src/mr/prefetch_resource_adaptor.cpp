/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

namespace RMM_NAMESPACE {
namespace mr {

prefetch_resource_adaptor::prefetch_resource_adaptor(device_async_resource_ref upstream)
  : shared_base(cuda::mr::make_shared_resource<detail::prefetch_resource_adaptor_impl>(upstream))
{
}

device_async_resource_ref prefetch_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

// Begin legacy device_memory_resource compatibility layer
void* prefetch_resource_adaptor::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void prefetch_resource_adaptor::do_deallocate(void* ptr,
                                              std::size_t bytes,
                                              cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool prefetch_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<prefetch_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
