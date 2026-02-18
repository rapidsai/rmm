/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

pool_memory_resource::pool_memory_resource(device_async_resource_ref upstream,
                                           std::size_t initial_pool_size,
                                           std::optional<std::size_t> maximum_pool_size)
  : shared_base(cuda::mr::make_shared_resource<detail::pool_memory_resource_impl>(
      upstream, initial_pool_size, maximum_pool_size))
{
}

device_async_resource_ref pool_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t pool_memory_resource::pool_size() const noexcept { return get().pool_size(); }

// Begin legacy device_memory_resource compatibility layer
void* pool_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void pool_memory_resource::do_deallocate(void* ptr,
                                         std::size_t bytes,
                                         cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool pool_memory_resource::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == &other) { return true; }
  auto const* cast = dynamic_cast<pool_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
