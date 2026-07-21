/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/arena_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

arena_memory_resource::arena_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> arena_size,
  bool dump_log_on_failure)
  : shared_base(cuda::mr::make_shared_resource<detail::arena_memory_resource_impl>(
      std::move(upstream), arena_size, dump_log_on_failure))
{
}

arena_memory_resource::~arena_memory_resource() = default;

void* arena_memory_resource::allocate(cuda::stream_ref stream,
                                      std::size_t bytes,
                                      std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void arena_memory_resource::deallocate(cuda::stream_ref stream,
                                       void* ptr,
                                       std::size_t bytes,
                                       std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* arena_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void arena_memory_resource::deallocate_sync(void* ptr,
                                            std::size_t bytes,
                                            std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool arena_memory_resource::operator==(arena_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
