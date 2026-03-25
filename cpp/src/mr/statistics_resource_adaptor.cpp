/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cstddef>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

device_async_resource_ref statistics_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

statistics_resource_adaptor::counter statistics_resource_adaptor::get_bytes_counter() const noexcept
{
  return get().get_bytes_counter();
}

statistics_resource_adaptor::counter statistics_resource_adaptor::get_allocations_counter()
  const noexcept
{
  return get().get_allocations_counter();
}

std::pair<statistics_resource_adaptor::counter, statistics_resource_adaptor::counter>
statistics_resource_adaptor::push_counters()
{
  return get().push_counters();
}

std::pair<statistics_resource_adaptor::counter, statistics_resource_adaptor::counter>
statistics_resource_adaptor::pop_counters()
{
  return get().pop_counters();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
