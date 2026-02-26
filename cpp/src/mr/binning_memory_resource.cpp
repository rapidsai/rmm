/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/binning_memory_resource.hpp>

#include <cstddef>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {

binning_memory_resource::binning_memory_resource(device_async_resource_ref upstream)
  : shared_base(cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(upstream))
{
}

binning_memory_resource::binning_memory_resource(device_async_resource_ref upstream,
                                                 int8_t min_size_exponent,
                                                 int8_t max_size_exponent)
  : shared_base(cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(
      upstream, min_size_exponent, max_size_exponent))
{
}

device_async_resource_ref binning_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

void binning_memory_resource::add_bin(std::size_t allocation_size,
                                      std::optional<device_async_resource_ref> bin_resource)
{
  get().add_bin(allocation_size, bin_resource);
}

// Begin legacy device_memory_resource compatibility layer
void* binning_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void binning_memory_resource::do_deallocate(void* ptr,
                                            std::size_t bytes,
                                            cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool binning_memory_resource::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<binning_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
