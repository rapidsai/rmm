/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cstddef>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

statistics_resource_adaptor::statistics_resource_adaptor(device_async_resource_ref upstream)
  : shared_base(cuda::mr::make_shared_resource<detail::statistics_resource_adaptor_impl>(upstream))
{
}

device_async_resource_ref statistics_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

statistics_resource_adaptor::counter statistics_resource_adaptor::get_bytes_counter() const noexcept
{
  return get().get_bytes_counter();
}

statistics_resource_adaptor::counter statistics_resource_adaptor::get_allocations_counter()
  const noexcept
{
  return get().get_allocations_counter();
}

std::pair<statistics_resource_adaptor::counter, statistics_resource_adaptor::counter>
statistics_resource_adaptor::push_counters()
{
  return get().push_counters();
}

std::pair<statistics_resource_adaptor::counter, statistics_resource_adaptor::counter>
statistics_resource_adaptor::pop_counters()
{
  return get().pop_counters();
}

// Begin legacy device_memory_resource compatibility layer
void* statistics_resource_adaptor::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void statistics_resource_adaptor::do_deallocate(void* ptr,
                                                std::size_t bytes,
                                                cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool statistics_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<statistics_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
