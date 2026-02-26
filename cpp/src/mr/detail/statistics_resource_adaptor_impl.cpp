/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/detail/statistics_resource_adaptor_impl.hpp>

#include <stdexcept>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

statistics_resource_adaptor_impl::statistics_resource_adaptor_impl(
  device_async_resource_ref upstream)
  : upstream_mr_{upstream}
{
}

device_async_resource_ref statistics_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

statistics_resource_adaptor_impl::counter statistics_resource_adaptor_impl::get_bytes_counter()
  const noexcept
{
  read_lock_t lock(mtx_);
  return counter_stack_.top().first;
}

statistics_resource_adaptor_impl::counter
statistics_resource_adaptor_impl::get_allocations_counter() const noexcept
{
  read_lock_t lock(mtx_);
  return counter_stack_.top().second;
}

std::pair<statistics_resource_adaptor_impl::counter, statistics_resource_adaptor_impl::counter>
statistics_resource_adaptor_impl::push_counters()
{
  write_lock_t lock(mtx_);
  auto ret = counter_stack_.top();
  counter_stack_.push({counter{}, counter{}});
  return ret;
}

std::pair<statistics_resource_adaptor_impl::counter, statistics_resource_adaptor_impl::counter>
statistics_resource_adaptor_impl::pop_counters()
{
  write_lock_t lock(mtx_);
  if (counter_stack_.size() < 2) { throw std::out_of_range("cannot pop the last counter pair"); }
  auto ret = counter_stack_.top();
  counter_stack_.pop();
  counter_stack_.top().first.add_counters_from_tracked_sub_block(ret.first);
  counter_stack_.top().second.add_counters_from_tracked_sub_block(ret.second);
  return ret;
}

void* statistics_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                                 std::size_t bytes,
                                                 std::size_t alignment)
{
  void* ptr = upstream_mr_.allocate(stream, bytes, alignment);
  {
    write_lock_t lock(mtx_);
    counter_stack_.top().first += bytes;
    counter_stack_.top().second += 1;
  }
  return ptr;
}

void statistics_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                                  void* ptr,
                                                  std::size_t bytes,
                                                  std::size_t alignment) noexcept
{
  upstream_mr_.deallocate(stream, ptr, bytes, alignment);
  {
    write_lock_t lock(mtx_);
    counter_stack_.top().first -= bytes;
    counter_stack_.top().second -= 1;
  }
}

void* statistics_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void statistics_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                       std::size_t bytes,
                                                       std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
