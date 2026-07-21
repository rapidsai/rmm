/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/sam_headroom_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

sam_headroom_memory_resource::sam_headroom_memory_resource(std::size_t headroom)
  : shared_base(cuda::mr::make_shared_resource<detail::sam_headroom_memory_resource_impl>(headroom))
{
}

sam_headroom_memory_resource::~sam_headroom_memory_resource() = default;

void* sam_headroom_memory_resource::allocate(cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void sam_headroom_memory_resource::deallocate(cuda::stream_ref stream,
                                              void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* sam_headroom_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void sam_headroom_memory_resource::deallocate_sync(void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool sam_headroom_memory_resource::operator==(
  sam_headroom_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
