/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/caching_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

caching_memory_resource::caching_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> max_split_size,
  caching_memory_resource_oom_fallback_policy oom_fallback_policy,
  caching_memory_resource_pool_policy pool_policy,
  caching_memory_resource_stream_reuse_policy stream_reuse_policy)
  : shared_base(cuda::mr::make_shared_resource<detail::caching_memory_resource_impl>(
      std::move(upstream), max_split_size, oom_fallback_policy, pool_policy, stream_reuse_policy))
{
}

device_async_resource_ref caching_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t caching_memory_resource::release() noexcept { return get().release(); }

std::size_t caching_memory_resource::release_small_blocks() noexcept
{
  return get().release_small_blocks();
}

std::size_t caching_memory_resource::release_large_blocks() noexcept
{
  return get().release_large_blocks();
}

std::size_t caching_memory_resource::cached_small_bytes() const noexcept
{
  return get().cached_small_bytes();
}

std::size_t caching_memory_resource::cached_large_bytes() const noexcept
{
  return get().cached_large_bytes();
}

std::size_t caching_memory_resource::upstream_allocation_size(std::size_t bytes) const noexcept
{
  return get().upstream_allocation_size(bytes);
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
