/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/prefetch_resource_adaptor.hpp>

namespace RMM_NAMESPACE {
namespace mr {

prefetch_resource_adaptor::prefetch_resource_adaptor(device_async_resource_ref upstream)
  : shared_base(cuda::mr::make_shared_resource<detail::prefetch_resource_adaptor_impl>(upstream))
{
}

device_async_resource_ref prefetch_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
