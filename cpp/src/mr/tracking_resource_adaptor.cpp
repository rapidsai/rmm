/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstddef>
#include <map>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {

tracking_resource_adaptor::tracking_resource_adaptor(device_async_resource_ref upstream,
                                                     bool capture_stacks)
  : shared_base(cuda::mr::make_shared_resource<detail::tracking_resource_adaptor_impl>(
      upstream, capture_stacks))
{
}

device_async_resource_ref tracking_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::map<void*, tracking_resource_adaptor::allocation_info> const&
tracking_resource_adaptor::get_outstanding_allocations() const noexcept
{
  return get().get_outstanding_allocations();
}

std::size_t tracking_resource_adaptor::get_allocated_bytes() const noexcept
{
  return get().get_allocated_bytes();
}

std::string tracking_resource_adaptor::get_outstanding_allocations_str() const
{
  return get().get_outstanding_allocations_str();
}

void tracking_resource_adaptor::log_outstanding_allocations() const
{
  get().log_outstanding_allocations();
}

// Begin legacy device_memory_resource compatibility layer
void* tracking_resource_adaptor::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void tracking_resource_adaptor::do_deallocate(void* ptr,
                                              std::size_t bytes,
                                              cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool tracking_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<tracking_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
