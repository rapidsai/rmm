/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/detail/limiting_resource_adaptor_impl.hpp>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

limiting_resource_adaptor_impl::limiting_resource_adaptor_impl(device_async_resource_ref upstream,
                                                               std::size_t allocation_limit,
                                                               std::size_t alignment)
  : upstream_mr_{upstream},
    allocation_limit_{allocation_limit},
    allocated_bytes_(0),
    alignment_(alignment)
{
}

device_async_resource_ref limiting_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t limiting_resource_adaptor_impl::get_allocated_bytes() const { return allocated_bytes_; }

std::size_t limiting_resource_adaptor_impl::get_allocation_limit() const
{
  return allocation_limit_;
}

void* limiting_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                               std::size_t bytes,
                                               std::size_t /*alignment*/)
{
  auto const proposed_size = rmm::align_up(bytes, alignment_);
  auto const old           = allocated_bytes_.fetch_add(proposed_size);
  if (old + proposed_size <= allocation_limit_) {
    try {
      return upstream_mr_.allocate(stream, bytes);
    } catch (...) {
      allocated_bytes_ -= proposed_size;
      throw;
    }
  }

  allocated_bytes_ -= proposed_size;
  auto const msg = std::string("Exceeded memory limit (failed to allocate ") +
                   rmm::detail::format_bytes(bytes) + ")";
  RMM_FAIL(msg.c_str(), rmm::out_of_memory);
}

void limiting_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                void* ptr,
                                                std::size_t bytes,
                                                std::size_t /*alignment*/) noexcept
{
  std::size_t const allocated_size = rmm::align_up(bytes, alignment_);
  upstream_mr_.deallocate(stream, ptr, bytes);
  allocated_bytes_ -= allocated_size;
}

void* limiting_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void limiting_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                     std::size_t bytes,
                                                     std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
