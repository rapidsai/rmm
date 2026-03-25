/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstddef>
#include <map>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {

device_async_resource_ref tracking_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::map<void*, tracking_resource_adaptor::allocation_info> const&
tracking_resource_adaptor::get_outstanding_allocations() const noexcept
{
  return get().get_outstanding_allocations();
}

std::size_t tracking_resource_adaptor::get_allocated_bytes() const noexcept
{
  return get().get_allocated_bytes();
}

std::string tracking_resource_adaptor::get_outstanding_allocations_str() const
{
  return get().get_outstanding_allocations_str();
}

void tracking_resource_adaptor::log_outstanding_allocations() const
{
  get().log_outstanding_allocations();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
