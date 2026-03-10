/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

namespace RMM_NAMESPACE {
namespace mr {

limiting_resource_adaptor::limiting_resource_adaptor(device_async_resource_ref upstream,
                                                     std::size_t allocation_limit,
                                                     std::size_t alignment)
  : shared_base(cuda::mr::make_shared_resource<detail::limiting_resource_adaptor_impl>(
      upstream, allocation_limit, alignment))
{
}

limiting_resource_adaptor::limiting_resource_adaptor(device_memory_resource* upstream,
                                                     std::size_t allocation_limit,
                                                     std::size_t alignment)
  : limiting_resource_adaptor(
      to_device_async_resource_ref_checked(upstream), allocation_limit, alignment)
{
}

device_async_resource_ref limiting_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t limiting_resource_adaptor::get_allocated_bytes() const
{
  return get().get_allocated_bytes();
}

std::size_t limiting_resource_adaptor::get_allocation_limit() const
{
  return get().get_allocation_limit();
}

// Begin legacy device_memory_resource compatibility layer
void* limiting_resource_adaptor::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void limiting_resource_adaptor::do_deallocate(void* ptr,
                                              std::size_t bytes,
                                              cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool limiting_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<limiting_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
