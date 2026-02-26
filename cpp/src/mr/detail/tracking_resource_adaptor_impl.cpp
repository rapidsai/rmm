/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/detail/tracking_resource_adaptor_impl.hpp>

#include <sstream>
#include <stdexcept>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

tracking_resource_adaptor_impl::tracking_resource_adaptor_impl(device_async_resource_ref upstream,
                                                               bool capture_stacks)
  : capture_stacks_{capture_stacks}, upstream_mr_{upstream}
{
}

device_async_resource_ref tracking_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::map<void*, tracking_resource_adaptor_impl::allocation_info> const&
tracking_resource_adaptor_impl::get_outstanding_allocations() const noexcept
{
  return allocations_;
}

std::size_t tracking_resource_adaptor_impl::get_allocated_bytes() const noexcept
{
  return allocated_bytes_;
}

std::string tracking_resource_adaptor_impl::get_outstanding_allocations_str() const
{
  read_lock_t lock(mtx_);
  std::ostringstream oss;
  for (auto const& alloc : allocations_) {
    oss << alloc.first << ": " << alloc.second.allocation_size << " B";
    if (alloc.second.strace != nullptr) {
      oss << " : callstack:" << std::endl << *alloc.second.strace;
    }
    oss << std::endl;
  }
  return oss.str();
}

void tracking_resource_adaptor_impl::log_outstanding_allocations() const
{
#if RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
  RMM_LOG_DEBUG("Outstanding Allocations: %s", get_outstanding_allocations_str());
#endif
}

void* tracking_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                               std::size_t bytes,
                                               std::size_t alignment)
{
  void* ptr = upstream_mr_.allocate(stream, bytes, alignment);
  {
    write_lock_t lock(mtx_);
    allocations_.emplace(ptr, allocation_info{bytes, capture_stacks_});
  }
  allocated_bytes_ += bytes;
  return ptr;
}

void tracking_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                void* ptr,
                                                std::size_t bytes,
                                                std::size_t alignment) noexcept
{
  upstream_mr_.deallocate(stream, ptr, bytes, alignment);
  {
    write_lock_t lock(mtx_);
    auto const found = allocations_.find(ptr);
    if (found == allocations_.end()) {
      RMM_LOG_ERROR(
        "Deallocating a pointer that was not tracked. Ptr: %p [%zuB], Current Num. Allocations: "
        "%zu",
        ptr,
        bytes,
        this->allocations_.size());
    } else {
      auto const allocated_bytes = found->second.allocation_size;
      allocations_.erase(found);
      if (allocated_bytes != bytes) {
        RMM_LOG_ERROR(
          "Alloc bytes (%zu) and Dealloc bytes (%zu) do not match", allocated_bytes, bytes);
        bytes = allocated_bytes;
      }
    }
  }
  allocated_bytes_ -= bytes;
}

void* tracking_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void tracking_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                     std::size_t bytes,
                                                     std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
