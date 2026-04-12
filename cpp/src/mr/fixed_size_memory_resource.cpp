/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/fixed_size_memory_resource.hpp>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {

fixed_size_memory_resource::fixed_size_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t block_size,
  std::size_t blocks_to_preallocate)
  : shared_base(cuda::mr::make_shared_resource<detail::fixed_size_memory_resource_impl>(
      std::move(upstream), block_size, blocks_to_preallocate))
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

}  // namespace mr
}  // namespace RMM_NAMESPACE
