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

}  // namespace mr
}  // namespace RMM_NAMESPACE
