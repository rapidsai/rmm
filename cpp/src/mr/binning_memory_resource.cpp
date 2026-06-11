/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/binning_memory_resource.hpp>

#include <cstddef>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {

binning_memory_resource::binning_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
  : shared_base(
      cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(std::move(upstream)))
{
}

binning_memory_resource::binning_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  int8_t min_size_exponent,
  int8_t max_size_exponent)
  : shared_base(cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(
      std::move(upstream), min_size_exponent, max_size_exponent))
{
}

binning_memory_resource::~binning_memory_resource() = default;

void* binning_memory_resource::allocate(cuda::stream_ref stream,
                                        std::size_t bytes,
                                        std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void binning_memory_resource::deallocate(cuda::stream_ref stream,
                                         void* ptr,
                                         std::size_t bytes,
                                         std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* binning_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void binning_memory_resource::deallocate_sync(void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

bool binning_memory_resource::operator==(binning_memory_resource const& other) const noexcept
{
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
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
