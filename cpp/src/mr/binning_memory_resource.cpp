/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/binning_memory_resource.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <optional>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

binning_memory_resource::binning_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_resource,
  int8_t min_size_exponent,
  int8_t max_size_exponent)
  : shared_base(cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(
      std::move(upstream_resource), min_size_exponent, max_size_exponent))
{
}

device_async_resource_ref binning_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

void binning_memory_resource::add_bin(std::size_t allocation_size,
                                      std::optional<device_async_resource_ref> bin_resource)
{
  get().add_bin(allocation_size, bin_resource);
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
