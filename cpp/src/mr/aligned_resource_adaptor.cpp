/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {

aligned_resource_adaptor::aligned_resource_adaptor(device_async_resource_ref upstream,
                                                   std::size_t alignment,
                                                   std::size_t alignment_threshold)
  : shared_base(cuda::mr::make_shared_resource<detail::aligned_resource_adaptor_impl>(
      upstream, alignment, alignment_threshold))
{
}

device_async_resource_ref aligned_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
