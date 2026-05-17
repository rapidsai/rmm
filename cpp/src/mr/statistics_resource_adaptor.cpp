/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cstddef>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

statistics_resource_adaptor::statistics_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
  : shared_base(
      cuda::mr::make_shared_resource<detail::statistics_resource_adaptor_impl>(std::move(upstream)))
{
}

statistics_resource_adaptor::~statistics_resource_adaptor() = default;

void* statistics_resource_adaptor::allocate(cuda::stream_ref stream,
                                            std::size_t bytes,
                                            std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void statistics_resource_adaptor::deallocate(cuda::stream_ref stream,
                                             void* ptr,
                                             std::size_t bytes,
                                             std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* statistics_resource_adaptor::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void statistics_resource_adaptor::deallocate_sync(void* ptr,
                                                  std::size_t bytes,
                                                  std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool statistics_resource_adaptor::operator==(
  statistics_resource_adaptor const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

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
