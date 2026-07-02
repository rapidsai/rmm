/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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

callback_memory_resource::~callback_memory_resource() = default;

void* callback_memory_resource::allocate(cuda::stream_ref stream,
                                         std::size_t bytes,
                                         std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void callback_memory_resource::deallocate(cuda::stream_ref stream,
                                          void* ptr,
                                          std::size_t bytes,
                                          std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* callback_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void callback_memory_resource::deallocate_sync(void* ptr,
                                               std::size_t bytes,
                                               std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool callback_memory_resource::operator==(callback_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
