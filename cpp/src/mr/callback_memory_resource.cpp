/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/detail/callback_memory_resource_impl.hpp>

#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

callback_memory_resource::callback_memory_resource(allocate_callback_t allocate_callback,
                                                   deallocate_callback_t deallocate_callback,
                                                   void* allocate_callback_arg,
                                                   void* deallocate_callback_arg)
  : shared_base(cuda::mr::make_shared_resource<detail::callback_memory_resource_impl>(
      std::move(allocate_callback),
      std::move(deallocate_callback),
      allocate_callback_arg,
      deallocate_callback_arg))
{
}

// Begin legacy device_memory_resource compatibility layer
void* callback_memory_resource::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void callback_memory_resource::do_deallocate(void* ptr,
                                             std::size_t bytes,
                                             cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool callback_memory_resource::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == std::addressof(other)) { return true; }
  auto const* cast = dynamic_cast<callback_memory_resource const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
