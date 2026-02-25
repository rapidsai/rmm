/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/detail/pool_memory_resource_impl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

#ifdef RMM_DEBUG_PRINT
#include <rmm/cuda_device.hpp>

#include <iostream>
#endif

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

pool_memory_resource_impl::pool_memory_resource_impl(device_async_resource_ref upstream,
                                                     std::size_t initial_pool_size,
                                                     std::optional<std::size_t> maximum_pool_size)
  : upstream_mr_{upstream}
{
  RMM_EXPECTS(rmm::is_aligned(initial_pool_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
              "Error, Initial pool size required to be a multiple of 256 bytes");
  RMM_EXPECTS(rmm::is_aligned(maximum_pool_size.value_or(0), rmm::CUDA_ALLOCATION_ALIGNMENT),
              "Error, Maximum pool size required to be a multiple of 256 bytes");

  initialize_pool(initial_pool_size, maximum_pool_size);
}

pool_memory_resource_impl::~pool_memory_resource_impl() { release(); }

device_async_resource_ref pool_memory_resource_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t pool_memory_resource_impl::pool_size() const noexcept { return current_pool_size_; }

std::size_t pool_memory_resource_impl::get_maximum_allocation_size() const
{
  return std::numeric_limits<std::size_t>::max();
}

pool_memory_resource_impl::block_type pool_memory_resource_impl::try_to_expand(
  std::size_t try_size, std::size_t min_size, cuda_stream_view stream)
{
  auto report_error = [&](const char* reason) {
    RMM_LOG_ERROR("[A][Stream %s][Upstream %zuB][FAILURE maximum pool size exceeded: %s]",
                  rmm::detail::format_stream(stream),
                  min_size,
                  reason);
    auto const msg = std::string("Maximum pool size exceeded (failed to allocate ") +
                     rmm::detail::format_bytes(min_size) + std::string("): ") + reason;
    RMM_FAIL(msg.c_str(), rmm::out_of_memory);
  };

  while (try_size >= min_size) {
    try {
      auto block = block_from_upstream(try_size, stream);
      current_pool_size_ += block.size();
      return block;
    } catch (std::exception const& e) {
      if (try_size == min_size) { report_error(e.what()); }
    }
    try_size = std::max(min_size, try_size / 2);
  }

  auto const max_size = maximum_pool_size_.value_or(std::numeric_limits<std::size_t>::max());
  auto const msg      = std::string("Not enough room to grow, current/max/try size = ") +
                   rmm::detail::format_bytes(pool_size()) + ", " +
                   rmm::detail::format_bytes(max_size) + ", " + rmm::detail::format_bytes(min_size);
  report_error(msg.c_str());
  return {};
}

void pool_memory_resource_impl::initialize_pool(std::size_t initial_size,
                                                std::optional<std::size_t> maximum_size)
{
  current_pool_size_ = 0;
  maximum_pool_size_ = maximum_size;

  RMM_EXPECTS(initial_size <= maximum_pool_size_.value_or(std::numeric_limits<std::size_t>::max()),
              "Initial pool size exceeds the maximum pool size!");

  if (initial_size > 0) {
    auto const block = try_to_expand(initial_size, initial_size, cuda_stream_legacy);
    this->insert_block(block, cuda_stream_legacy);
  }
}

pool_memory_resource_impl::block_type pool_memory_resource_impl::expand_pool(
  std::size_t size, [[maybe_unused]] free_list& blocks, cuda_stream_view stream)
{
  return try_to_expand(size_to_grow(size), size, stream);
}

std::size_t pool_memory_resource_impl::size_to_grow(std::size_t size) const
{
  if (maximum_pool_size_.has_value()) {
    auto const unaligned_remaining = maximum_pool_size_.value() - pool_size();
    auto const remaining    = rmm::align_up(unaligned_remaining, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto const aligned_size = rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    return (aligned_size <= remaining) ? std::max(aligned_size, remaining / 2) : 0;
  }
  return std::max(size, pool_size());
}

pool_memory_resource_impl::block_type pool_memory_resource_impl::block_from_upstream(
  std::size_t size, cuda_stream_view stream)
{
  RMM_LOG_DEBUG("[A][Stream %s][Upstream %zuB]", rmm::detail::format_stream(stream), size);

  if (size == 0) { return {}; }

  void* ptr = get_upstream_resource().allocate(stream, size);
  return *upstream_blocks_.emplace(static_cast<char*>(ptr), size, true).first;
}

pool_memory_resource_impl::split_block pool_memory_resource_impl::allocate_from_block(
  block_type const& block, std::size_t size)
{
  block_type const alloc{block.pointer(), size, block.is_head()};
#ifdef RMM_POOL_TRACK_ALLOCATIONS
  allocated_blocks_.insert(alloc);
#endif

  auto rest = (block.size() > size)
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                ? block_type{block.pointer() + size, block.size() - size, false}
                : block_type{};
  return {alloc, rest};
}

pool_memory_resource_impl::block_type pool_memory_resource_impl::free_block(
  void* ptr, std::size_t size) noexcept
{
#ifdef RMM_POOL_TRACK_ALLOCATIONS
  if (ptr == nullptr) return block_type{};
  auto const iter = allocated_blocks_.find(static_cast<char*>(ptr));
  RMM_LOGGING_ASSERT(iter != allocated_blocks_.end());

  auto block = *iter;
  RMM_LOGGING_ASSERT(block.size() == rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT));
  allocated_blocks_.erase(iter);

  return block;
#else
  auto const iter = upstream_blocks_.find(static_cast<char*>(ptr));
  return block_type{static_cast<char*>(ptr), size, (iter != upstream_blocks_.end())};
#endif
}

void pool_memory_resource_impl::release()
{
  lock_guard lock(this->get_mutex());

  for (auto block : upstream_blocks_) {
    get_upstream_resource().deallocate_sync(block.pointer(), block.size());
  }
  upstream_blocks_.clear();
#ifdef RMM_POOL_TRACK_ALLOCATIONS
  allocated_blocks_.clear();
#endif

  current_pool_size_ = 0;
}

std::pair<std::size_t, std::size_t> pool_memory_resource_impl::free_list_summary(
  free_list const& blocks)
{
  std::size_t largest{};
  std::size_t total{};
  std::for_each(blocks.cbegin(), blocks.cend(), [&largest, &total](auto const& block) {
    total += block.size();
    largest = std::max(largest, block.size());
  });
  return {largest, total};
}

#ifdef RMM_DEBUG_PRINT
void pool_memory_resource_impl::print()
{
  lock_guard lock(this->get_mutex());

  auto const [free, total] = rmm::available_device_memory();
  std::cout << "GPU free memory: " << free << " total: " << total << "\n";

  std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
  std::size_t upstream_total{0};

  for (auto blocks : upstream_blocks_) {
    blocks.print();
    upstream_total += blocks.size();
  }
  std::cout << "total upstream: " << upstream_total << " B\n";

#ifdef RMM_POOL_TRACK_ALLOCATIONS
  std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
  for (auto block : allocated_blocks_)
    block.print();
#endif

  this->print_free_blocks();
}
#endif

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
