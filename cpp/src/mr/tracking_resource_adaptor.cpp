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

tracking_resource_adaptor::tracking_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream, bool capture_stacks)
  : shared_base(cuda::mr::make_shared_resource<detail::tracking_resource_adaptor_impl>(
      std::move(upstream), capture_stacks))
{
}

tracking_resource_adaptor::~tracking_resource_adaptor() = default;

void* tracking_resource_adaptor::allocate(cuda::stream_ref stream,
                                          std::size_t bytes,
                                          std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void tracking_resource_adaptor::deallocate(cuda::stream_ref stream,
                                           void* ptr,
                                           std::size_t bytes,
                                           std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* tracking_resource_adaptor::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void tracking_resource_adaptor::deallocate_sync(void* ptr,
                                                std::size_t bytes,
                                                std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool tracking_resource_adaptor::operator==(tracking_resource_adaptor const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

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
