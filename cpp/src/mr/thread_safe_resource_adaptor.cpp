/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/thread_safe_resource_adaptor.hpp>

namespace RMM_NAMESPACE {
namespace mr {

device_async_resource_ref thread_safe_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
