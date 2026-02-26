/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/aligned_resource_adaptor_impl.hpp>

#include <algorithm>
#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

aligned_resource_adaptor_impl::aligned_resource_adaptor_impl(device_async_resource_ref upstream,
                                                             std::size_t alignment,
                                                             std::size_t alignment_threshold)
  : upstream_mr_{upstream},
    alignment_{std::max(alignment, rmm::CUDA_ALLOCATION_ALIGNMENT)},
    alignment_threshold_{alignment_threshold}
{
  RMM_EXPECTS(rmm::is_supported_alignment(alignment), "Allocation alignment is not a power of 2.");
}

device_async_resource_ref aligned_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t aligned_resource_adaptor_impl::get_alignment() const noexcept { return alignment_; }

std::size_t aligned_resource_adaptor_impl::get_alignment_threshold() const noexcept
{
  return alignment_threshold_;
}

std::size_t aligned_resource_adaptor_impl::upstream_allocation_size(std::size_t bytes) const
{
  auto const aligned_size = rmm::align_up(bytes, alignment_);
  return aligned_size + (alignment_ - rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void* aligned_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                              std::size_t bytes,
                                              std::size_t /*alignment*/)
{
  if (alignment_ == rmm::CUDA_ALLOCATION_ALIGNMENT || bytes < alignment_threshold_) {
    return upstream_mr_.allocate(stream, bytes, 1);
  }
  auto const size = upstream_allocation_size(bytes);
  void* pointer   = upstream_mr_.allocate(stream, size, 1);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto const address         = reinterpret_cast<std::size_t>(pointer);
  auto const aligned_address = rmm::align_up(address, alignment_);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
  void* aligned_pointer = reinterpret_cast<void*>(aligned_address);
  if (pointer != aligned_pointer) {
    std::lock_guard<std::mutex> lock(mtx_);
    pointers_.emplace(aligned_pointer, pointer);
  }
  return aligned_pointer;
}

void aligned_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                               void* ptr,
                                               std::size_t bytes,
                                               std::size_t /*alignment*/) noexcept
{
  if (alignment_ == rmm::CUDA_ALLOCATION_ALIGNMENT || bytes < alignment_threshold_) {
    upstream_mr_.deallocate(stream, ptr, bytes, 1);
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto const iter = pointers_.find(ptr);
      if (iter != pointers_.end()) {
        ptr = iter->second;
        pointers_.erase(iter);
      }
    }
    upstream_mr_.deallocate(stream, ptr, upstream_allocation_size(bytes), 1);
  }
}

void* aligned_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void aligned_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
