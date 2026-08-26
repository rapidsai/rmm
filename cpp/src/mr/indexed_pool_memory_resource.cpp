/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/indexed_pool_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

RMM_NAMESPACE_BEGIN
namespace mr {

indexed_pool_memory_resource::indexed_pool_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t initial_pool_size,
  std::optional<std::size_t> maximum_pool_size)
  : shared_base(cuda::mr::make_shared_resource<
                detail::pool_memory_resource_impl<detail::indexed_coalescing_free_list>>(
      std::move(upstream), initial_pool_size, maximum_pool_size))
{
}

device_async_resource_ref indexed_pool_memory_resource::get_upstream_resource() const noexcept
{ return get().get_upstream_resource(); }

std::size_t indexed_pool_memory_resource::pool_size() const noexcept { return get().pool_size(); }

}  // namespace mr
RMM_NAMESPACE_END
