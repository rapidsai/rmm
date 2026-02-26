/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {

fixed_size_memory_resource::fixed_size_memory_resource(device_async_resource_ref upstream,
                                                       std::size_t block_size,
                                                       std::size_t blocks_to_preallocate)
  : shared_base(cuda::mr::make_shared_resource<detail::fixed_size_memory_resource_impl>(
      upstream, block_size, blocks_to_preallocate))
{
}

device_async_resource_ref fixed_size_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t fixed_size_memory_resource::get_block_size() const noexcept
{
  return get().get_block_size();
}

// Begin legacy device_memory_resource compatibility layer
void* fixed_size_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void fixed_size_memory_resource::do_deallocate(void* ptr,
                                               std::size_t bytes,
                                               cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool fixed_size_memory_resource::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<fixed_size_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
