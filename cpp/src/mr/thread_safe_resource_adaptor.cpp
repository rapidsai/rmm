/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/thread_safe_resource_adaptor.hpp>

namespace RMM_NAMESPACE {
namespace mr {

thread_safe_resource_adaptor::thread_safe_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
  : shared_base(cuda::mr::make_shared_resource<detail::thread_safe_resource_adaptor_impl>(
      std::move(upstream)))
{
}

thread_safe_resource_adaptor::~thread_safe_resource_adaptor() = default;

void* thread_safe_resource_adaptor::allocate(cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void thread_safe_resource_adaptor::deallocate(cuda::stream_ref stream,
                                              void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* thread_safe_resource_adaptor::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void thread_safe_resource_adaptor::deallocate_sync(void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool thread_safe_resource_adaptor::operator==(
  thread_safe_resource_adaptor const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
}

device_async_resource_ref thread_safe_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
