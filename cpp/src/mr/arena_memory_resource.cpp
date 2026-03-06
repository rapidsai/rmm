/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/arena_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

arena_memory_resource::arena_memory_resource(device_async_resource_ref upstream_mr,
                                             std::optional<std::size_t> arena_size,
                                             bool dump_log_on_failure)
  : shared_base(cuda::mr::make_shared_resource<detail::arena_memory_resource_impl>(
      upstream_mr, arena_size, dump_log_on_failure))
{
}

// Begin legacy device_memory_resource compatibility layer
void* arena_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void arena_memory_resource::do_deallocate(void* ptr,
                                          std::size_t bytes,
                                          cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool arena_memory_resource::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<arena_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
