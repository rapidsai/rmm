/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/limiting_resource_adaptor.hpp>

namespace RMM_NAMESPACE {
namespace mr {

limiting_resource_adaptor::limiting_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t allocation_limit,
  std::size_t alignment)
  : shared_base(cuda::mr::make_shared_resource<detail::limiting_resource_adaptor_impl>(
      std::move(upstream), allocation_limit, alignment))
{
}

limiting_resource_adaptor::~limiting_resource_adaptor() = default;

void* limiting_resource_adaptor::allocate(cuda::stream_ref stream,
                                          std::size_t bytes,
                                          std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void limiting_resource_adaptor::deallocate(cuda::stream_ref stream,
                                           void* ptr,
                                           std::size_t bytes,
                                           std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* limiting_resource_adaptor::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void limiting_resource_adaptor::deallocate_sync(void* ptr,
                                                std::size_t bytes,
                                                std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool limiting_resource_adaptor::operator==(limiting_resource_adaptor const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

device_async_resource_ref limiting_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t limiting_resource_adaptor::get_allocated_bytes() const
{
  return get().get_allocated_bytes();
}

std::size_t limiting_resource_adaptor::get_allocation_limit() const
{
  return get().get_allocation_limit();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
