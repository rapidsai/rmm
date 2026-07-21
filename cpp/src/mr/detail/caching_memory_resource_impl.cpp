/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/detail/caching_memory_resource_impl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

caching_memory_resource_impl::caching_memory_resource_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> max_split_size,
  caching_memory_resource_oom_fallback_policy oom_fallback_policy,
  caching_memory_resource_pool_policy pool_policy,
  caching_memory_resource_stream_reuse_policy stream_reuse_policy)
  : upstream_mr_{std::move(upstream)},
    max_split_size_{max_split_size},
    oom_fallback_policy_{oom_fallback_policy},
    pool_policy_{pool_policy},
    stream_reuse_policy_{stream_reuse_policy}
{
}

caching_memory_resource_impl::~caching_memory_resource_impl() { (void)release(); }

device_async_resource_ref caching_memory_resource_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t caching_memory_resource_impl::release() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(true) + release_pool(false);
}

std::size_t caching_memory_resource_impl::release_small_blocks() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(true);
}

std::size_t caching_memory_resource_impl::release_large_blocks() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(false);
}

std::size_t caching_memory_resource_impl::cached_small_bytes() const noexcept
{
  return cached_small_bytes_;
}

std::size_t caching_memory_resource_impl::cached_large_bytes() const noexcept
{
  return cached_large_bytes_;
}

std::size_t caching_memory_resource_impl::upstream_allocation_size(std::size_t bytes) const noexcept
{
  if (bytes == 0) { return 0; }
  if (bytes <= small_size_threshold) { return small_segment_size; }
  if (bytes < minimum_large_allocation) { return large_segment_size; }
  return rmm::align_up(bytes, large_rounding);
}

bool caching_memory_resource_impl::supports_cross_stream_reuse() const noexcept
{
  return stream_reuse_policy_ == caching_memory_resource_stream_reuse_policy::cross_stream;
}

std::size_t caching_memory_resource_impl::get_maximum_allocation_size() const
{
  return std::numeric_limits<std::size_t>::max();
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::expand_pool(
  std::size_t size, free_list&, cuda_stream_view stream)
{
  auto const is_small      = is_small_allocation(size);
  auto const upstream_size = upstream_allocation_size(size);

  try {
    return block_from_upstream(upstream_size, stream, is_small);
  } catch (rmm::out_of_memory const&) {
    if (oom_fallback_policy_ == caching_memory_resource_oom_fallback_policy::release_all) {
      auto const released = release_all_pools();
      if (released == 0) { throw; }
      return block_from_upstream(upstream_size, stream, is_small);
    }

    if (release_oversized_blocks(upstream_size) != 0) {
      auto const block = [&]() {
        try {
          return block_from_upstream(upstream_size, stream, is_small);
        } catch (rmm::out_of_memory const&) {
          return block_type{};
        }
      }();
      if (block.is_valid()) { return block; }
    }

    auto const released = release_all_pools();
    if (released == 0) { throw; }
    return block_from_upstream(upstream_size, stream, is_small);
  }
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::get_block_from_free_list(
  free_list& blocks, std::size_t size)
{
  auto const max_split_size = max_split_size_.value_or(std::numeric_limits<std::size_t>::max());
  return blocks.get_block(size, [this, size, max_split_size](block_type const& block) {
    auto const block_is_small_pool = block_is_small(block);
    if (pool_policy_ == caching_memory_resource_pool_policy::separate &&
        block_is_small_pool != is_small_allocation(size)) {
      return false;
    }
    if (block_is_small_pool) { return true; }
    if (size < max_split_size && block.size() >= max_split_size) { return false; }
    return size < max_split_size || block.size() < size + large_segment_size;
  });
}

caching_memory_resource_impl::split_block caching_memory_resource_impl::allocate_from_block(
  block_type const& block, std::size_t size)
{
  auto const split = should_split(block, size);
  auto const alloc = block_type{block.pointer(), split ? size : block.size(), block.is_head()};
  allocated_blocks_.insert(alloc);
  if (!split) { return {alloc, {}}; }

  auto rest = block_type{block.pointer() + size, block.size() - size, false};
  return {alloc, rest};
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::free_block(
  void* ptr, std::size_t size) noexcept
{
  auto const iter = allocated_blocks_.find(static_cast<char*>(ptr));
  if (iter != allocated_blocks_.end()) {
    auto block = *iter;
    allocated_blocks_.erase(iter);
    return block;
  }

  auto const aligned_size = rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const upstream_iter =
    upstream_blocks_.upper_bound(segment{static_cast<char*>(ptr), 0, false});
  if (upstream_iter != upstream_blocks_.begin()) {
    auto const candidate   = std::prev(upstream_iter);
    auto const segment_end = candidate->pointer + candidate->size;
    if (candidate->pointer <= static_cast<char*>(ptr) && static_cast<char*>(ptr) < segment_end) {
      return block_type{
        static_cast<char*>(ptr), aligned_size, candidate->pointer == static_cast<char*>(ptr)};
    }
  }

  return block_type{static_cast<char*>(ptr), aligned_size, false};
}

std::pair<std::size_t, std::size_t> caching_memory_resource_impl::free_list_summary(
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

bool caching_memory_resource_impl::is_small_allocation(std::size_t size) const noexcept
{
  return size <= small_size_threshold;
}

bool caching_memory_resource_impl::should_split(block_type const& block,
                                                std::size_t size) const noexcept
{
  auto const remaining = block.size() - size;
  if (block_is_small(block)) { return remaining >= minimum_block_size; }
  auto const max_split_size = max_split_size_.value_or(std::numeric_limits<std::size_t>::max());
  return size < max_split_size && remaining > small_size_threshold;
}

bool caching_memory_resource_impl::block_is_small(block_type const& block) const noexcept
{
  auto const iter = upstream_blocks_.upper_bound(segment{block.pointer(), 0, false});
  if (iter == upstream_blocks_.begin()) { return block.size() <= small_size_threshold; }

  auto const candidate   = std::prev(iter);
  auto const segment_end = candidate->pointer + candidate->size;
  if (candidate->pointer <= block.pointer() && block.pointer() < segment_end) {
    return candidate->is_small;
  }

  return block.size() <= small_size_threshold;
}

std::size_t caching_memory_resource_impl::release_pool(bool is_small) noexcept
{
  this->merge_all_free_blocks();

  std::size_t released{};

  for (auto it = upstream_blocks_.begin(); it != upstream_blocks_.end();) {
    if (it->is_small != is_small) {
      ++it;
      continue;
    }

    auto const block = block_type{it->pointer, it->size, true};
    if (!this->remove_free_block(block)) {
      ++it;
      continue;
    }

    auto const size = it->size;
    get_upstream_resource().deallocate_sync(it->pointer, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    released += size;
    if (is_small) {
      cached_small_bytes_ -= size;
    } else {
      cached_large_bytes_ -= size;
    }
    it = upstream_blocks_.erase(it);
  }

  return released;
}

std::size_t caching_memory_resource_impl::release_all_pools() noexcept
{
  return release_pool(true) + release_pool(false);
}

std::size_t caching_memory_resource_impl::release_oversized_blocks(std::size_t size) noexcept
{
  auto const max_split_size = max_split_size_.value_or(std::numeric_limits<std::size_t>::max());
  if (max_split_size == std::numeric_limits<std::size_t>::max()) { return 0; }

  this->merge_all_free_blocks();

  auto const target_size = std::max(size, max_split_size);
  std::size_t released{};

  for (auto it = upstream_blocks_.begin();
       it != upstream_blocks_.end() && released < target_size;) {
    if (it->size < max_split_size) {
      ++it;
      continue;
    }

    auto const block = block_type{it->pointer, it->size, true};
    if (!this->remove_free_block(block)) {
      ++it;
      continue;
    }

    auto const block_size = it->size;
    get_upstream_resource().deallocate_sync(
      it->pointer, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    released += block_size;
    if (it->is_small) {
      cached_small_bytes_ -= block_size;
    } else {
      cached_large_bytes_ -= block_size;
    }
    it = upstream_blocks_.erase(it);
  }

  return released;
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::block_from_upstream(
  std::size_t size, cuda_stream_view stream, bool is_small)
{
  void* ptr = get_upstream_resource().allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  upstream_blocks_.insert(segment{static_cast<char*>(ptr), size, is_small});
  if (is_small) {
    cached_small_bytes_ += size;
  } else {
    cached_large_bytes_ += size;
  }
  return block_type{static_cast<char*>(ptr), size, true};
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
