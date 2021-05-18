/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/detail/coalescing_free_list.hpp>
#include <rmm/mr/device/detail/stream_ordered_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rmm {
namespace mr {

/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class pool_memory_resource final
  : public detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                  detail::coalescing_free_list> {
 public:
  friend class detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                      detail::coalescing_free_list>;

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream_mr`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   * @throws rmm::logic_error if `initial_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   * @throws rmm::logic_error if `maximum_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool. Defaults to half of the
   * available memory on the current device.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all
   * of the available memory on the current device.
   */
  explicit pool_memory_resource(Upstream* upstream_mr,
                                thrust::optional<std::size_t> initial_pool_size = thrust::nullopt,
                                thrust::optional<std::size_t> maximum_pool_size = thrust::nullopt)
    : upstream_mr_{[upstream_mr]() {
        RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");
        return upstream_mr;
      }()}
  {
    RMM_EXPECTS(rmm::detail::is_aligned(initial_pool_size.value_or(0),
                                        rmm::detail::CUDA_ALLOCATION_ALIGNMENT),
                "Error, Initial pool size required to be a multiple of 256 bytes");
    RMM_EXPECTS(rmm::detail::is_aligned(maximum_pool_size.value_or(0),
                                        rmm::detail::CUDA_ALLOCATION_ALIGNMENT),
                "Error, Maximum pool size required to be a multiple of 256 bytes");

    initialize_pool(initial_pool_size, maximum_pool_size);
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() { release(); }

  pool_memory_resource()                            = delete;
  pool_memory_resource(pool_memory_resource const&) = delete;
  pool_memory_resource(pool_memory_resource&&)      = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource& operator=(pool_memory_resource&&) = delete;

  /**
   * @brief Queries whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool true.
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool false
   */
  bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

 protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `size_t`
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  /**
   * @brief Try to expand the pool by allocating a block of at least `min_size` bytes from
   * upstream
   *
   * Attempts to allocate `try_size` bytes from upstream. If it fails, it iteratively reduces the
   * attempted size by half until `min_size`, returning the allocated block once it succeeds.
   *
   * @throws rmm::bad_alloc if `min_size` bytes cannot be allocated from upstream or maximum pool
   * size is exceeded.
   *
   * @param try_size The initial requested size to try allocating.
   * @param min_size The minimum requested size to try allocating.
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `min_size` bytes
   */
  block_type try_to_expand(std::size_t try_size, std::size_t min_size, cuda_stream_view stream)
  {
    while (try_size >= min_size) {
      auto b = block_from_upstream(try_size, stream);
      if (b.has_value()) {
        current_pool_size_ += b.value().size();
        return b.value();
      }
      if (try_size == min_size) break;  // only try `size` once
      try_size = std::max(min_size, try_size / 2);
    }
    RMM_LOG_ERROR("[A][Stream {}][Upstream {}B][FAILURE maximum pool size exceeded]",
                  fmt::ptr(stream.value()),
                  min_size);
    RMM_FAIL("Maximum pool size exceeded", rmm::bad_alloc);
  }

  /**
   * @brief Allocate initial memory for the pool
   *
   * If initial_size is unset, then queries the upstream memory resource for available memory if
   * upstream supports `get_mem_info`, or queries the device (using CUDA API) for available memory
   * if not. Then attempts to initialize to half the available memory.
   *
   * If initial_size is set, then tries to initialize the pool to that size.
   *
   * @param initial_size The optional initial size for the pool
   * @param maximum_size The optional maximum size for the pool
   */
  void initialize_pool(thrust::optional<std::size_t> initial_size,
                       thrust::optional<std::size_t> maximum_size)
  {
    auto const try_size = [&]() {
      if (not initial_size.has_value()) {
        std::size_t free{}, total{};
        std::tie(free, total) = (get_upstream()->supports_get_mem_info())
                                  ? get_upstream()->get_mem_info(cuda_stream_legacy)
                                  : rmm::detail::available_device_memory();
        return rmm::detail::align_up(std::min(free, total / 2),
                                     rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      } else {
        return initial_size.value();
      }
    }();

    current_pool_size_ = 0;  // try_to_expand will set this if it succeeds
    maximum_pool_size_ = maximum_size;

    RMM_EXPECTS(try_size <= maximum_pool_size_.value_or(std::numeric_limits<std::size_t>::max()),
                "Initial pool size exceeds the maximum pool size!");

    if (try_size > 0) {
      auto const b = try_to_expand(try_size, try_size, cuda_stream_legacy);
      this->insert_block(b, cuda_stream_legacy);
    }
  }

  /**
   * @brief Allocate space from upstream to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * @param size The minimum size to allocate
   * @param blocks The free list (ignored in this implementation)
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)
  {
    // Strategy: If maximum_pool_size_ is set, then grow geometrically, e.g. by halfway to the
    // limit each time. If it is not set, grow exponentially, e.g. by doubling the pool size each
    // time. Upon failure, attempt to back off exponentially, e.g. by half the attempted size,
    // until either success or the attempt is less than the requested size.
    return try_to_expand(size_to_grow(size), size, stream);
  }

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   *
   * Strategy is to try to grow the pool by half the difference between the configured maximum
   * pool size and the current pool size, if the maximum pool size is set. If it is not set, try
   * to double the current pool size.
   *
   * Returns 0 if the requested size cannot be satisfied.
   *
   * @param size The size of the minimum allocation immediately needed
   * @return size_t The computed size to grow the pool.
   */
  std::size_t size_to_grow(std::size_t size) const
  {
    if (maximum_pool_size_.has_value()) {
      auto const unaligned_remaining = maximum_pool_size_.value() - pool_size();
      auto const remaining =
        rmm::detail::align_up(unaligned_remaining, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      auto const aligned_size = rmm::detail::align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      return (aligned_size <= remaining) ? std::max(aligned_size, remaining / 2) : 0;
    } else
      return std::max(size, pool_size());
  };

  /**
   * @brief Allocate a block from upstream to expand the suballocation pool.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  thrust::optional<block_type> block_from_upstream(size_t size, cuda_stream_view stream)
  {
    RMM_LOG_DEBUG("[A][Stream {}][Upstream {}B]", fmt::ptr(stream.value()), size);

    if (size == 0) return {};

    try {
      void* p = upstream_mr_->allocate(size, stream);
      return thrust::optional<block_type>{
        *upstream_blocks_.emplace(reinterpret_cast<char*>(p), size, true).first};
    } catch (std::exception const& e) {
      return thrust::nullopt;
    }
  }

  /**
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& b, size_t size)
  {
    block_type const alloc{b.pointer(), size, b.is_head()};
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    allocated_blocks_.insert(alloc);
#endif

    auto rest =
      (b.size() > size) ? block_type{b.pointer() + size, b.size() - size, false} : block_type{};
    return {reinterpret_cast<void*>(alloc.pointer()), rest};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @param stream The stream-event pair for the stream on which the memory was last used.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* p, size_t size) noexcept
  {
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    if (p == nullptr) return block_type{};
    auto const i = allocated_blocks_.find(static_cast<char*>(p));
    RMM_LOGGING_ASSERT(i != allocated_blocks_.end());

    auto block = *i;
    RMM_LOGGING_ASSERT(block.size() == rmm::detail::align_up(size, allocation_alignment));
    allocated_blocks_.erase(i);

    return block;
#else
    auto const i = upstream_blocks_.find(static_cast<char*>(p));
    return block_type{static_cast<char*>(p), size, (i != upstream_blocks_.end())};
#endif
  }

  /**
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return size_t The total size of the currently allocated pool.
   */
  size_t pool_size() const noexcept { return current_pool_size_; }

  /**
   * @brief Free all memory allocated from the upstream memory_resource.
   *
   */
  void release()
  {
    lock_guard lock(this->get_mutex());

    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.pointer(), b.size());
    upstream_blocks_.clear();
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    allocated_blocks_.clear();
#endif

    current_pool_size_ = 0;
  }

  /**
   * @brief Print debugging information about all blocks in the pool.
   *
   * @note This function is intended only for use in debugging.
   *
   */
  void print()
  {
    lock_guard lock(this->get_mutex());

    std::size_t free, total;
    std::tie(free, total) = upstream_mr_->get_mem_info(0);
    std::cout << "GPU free memory: " << free << " total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto h : upstream_blocks_) {
      h.print();
      upstream_total += h.size();
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

#ifdef RMM_POOL_TRACK_ALLOCATIONS
    std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
    for (auto b : allocated_blocks_)
      b.print();
#endif

    this->print_free_blocks();
  }

  /**
   * @brief Get the largest available block size and total free size in the specified free list
   *
   * This is intended only for debugging
   *
   * @param blocks The free list from which to return the summary
   * @return std::pair<std::size_t, std::size_t> Pair of largest available block, total free size
   */
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks)
  {
    std::size_t largest{};
    std::size_t total{};
    std::for_each(blocks.cbegin(), blocks.cend(), [&largest, &total](auto const& b) {
      total += b.size();
      largest = std::max(largest, b.size());
    });
    return {largest, total};
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

  Upstream* upstream_mr_;  // The "heap" to allocate the pool from
  std::size_t current_pool_size_{};
  thrust::optional<std::size_t> maximum_pool_size_{};

#ifdef RMM_POOL_TRACK_ALLOCATIONS
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> allocated_blocks_;
#endif

  // blocks allocated from upstream
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;
};  // namespace mr

}  // namespace mr
}  // namespace rmm
