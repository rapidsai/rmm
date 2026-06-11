/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {

pool_memory_resource::pool_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t initial_pool_size,
  std::optional<std::size_t> maximum_pool_size)
  : shared_base(cuda::mr::make_shared_resource<detail::pool_memory_resource_impl>(
      std::move(upstream), initial_pool_size, maximum_pool_size))
{
}

pool_memory_resource::~pool_memory_resource() = default;

void* pool_memory_resource::allocate(cuda::stream_ref stream,
                                     std::size_t bytes,
                                     std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void pool_memory_resource::deallocate(cuda::stream_ref stream,
                                      void* ptr,
                                      std::size_t bytes,
                                      std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* pool_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void pool_memory_resource::deallocate_sync(void* ptr,
                                           std::size_t bytes,
                                           std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool pool_memory_resource::operator==(pool_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

device_async_resource_ref pool_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t pool_memory_resource::pool_size() const noexcept { return get().pool_size(); }

}  // namespace mr
}  // namespace RMM_NAMESPACE
