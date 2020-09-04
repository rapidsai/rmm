/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/detail/arena.hpp>
#include <rmm/mr/device/detail/coalescing_free_list.hpp>
#include <rmm/mr/device/detail/free_list.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rmm {
namespace mr {

/**
 * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena_memory_resource final : public device_memory_resource {
 public:
  static constexpr size_t allocation_alignment        = 256;
  static constexpr size_t minimum_upstream_block_size = 1UL << 22UL;  // 4 MiB

  /**
   * @brief Construct a `arena_memory_resource`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   */
  explicit arena_memory_resource(Upstream* upstream_mr) : upstream_mr_{upstream_mr} {}

  /**
   * @brief Destroy the `arena_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~arena_memory_resource() override { release(); }

  arena_memory_resource()                             = delete;
  arena_memory_resource(arena_memory_resource const&) = delete;
  arena_memory_resource(arena_memory_resource&&)      = delete;
  arena_memory_resource& operator=(arena_memory_resource const&) = delete;
  arena_memory_resource& operator=(arena_memory_resource&&) = delete;

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

 private:
  using arena            = detail::arena;
  using free_list        = detail::arena::free_list;
  using block_type       = detail::arena::block_type;
  using split_block_type = detail::arena::split_block_type;
  using allocated_set    = detail::arena::allocated_set;
  using lock_guard       = std::lock_guard<std::mutex>;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource.
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `size_t`.
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  /**
   * @brief Allocate a block from upstream to expand the arena.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @return block_type The allocated block
   */
  block_type block_from_upstream(size_t size)
  {
    void* p = upstream_mr_->allocate(size, cudaStreamLegacy);
    block_type b{reinterpret_cast<char*>(p), size, true, size};
    return b;
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a sufficiently sized block.
   *
   * @param size The minimum size to allocate
   * @param blocks The free list
   * @return block_type a block of at least `size` bytes
   */
  block_type expand_arena(size_t size)
  {
    auto grow_size = std::max(size, minimum_upstream_block_size);
    return block_from_upstream(grow_size);
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(free_list& free_blocks, size_t size)
  {
    // Try to find a satisfactory block in free list for the current arena (no sync required)
    block_type b = free_blocks.get_block(size);
    if (b.is_valid()) return b;

    // no larger blocks available, so grow the arena and create a block
    return expand_arena(size);
  }

  /**
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the arena.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block_type allocate_from_block(block_type const& b,
                                       size_t size,
                                       allocated_set& allocated_blocks)
  {
    block_type const alloc{b.pointer(), size, b.is_head(), b.original_size()};
    allocated_blocks.insert(alloc);

    auto rest =
      (b.size() > size) ? block_type{b.pointer() + size, b.size() - size, false, 0} : block_type{};
    return {reinterpret_cast<void*>(alloc.pointer()), rest};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* p, size_t size, allocated_set& allocated_blocks) noexcept
  {
    auto const i = allocated_blocks.find(static_cast<char*>(p));

    // The pointer may be allocated in another arena.
    if (i == allocated_blocks.end()) { return {}; }

    auto block = *i;
    assert(block.size() == size);
    allocated_blocks.erase(i);

    return block;
  }

  void shrink_arena(free_list& free_blocks, cudaStream_t stream)
  {
    bool synchronized = false;
    for (auto it = std::next(free_blocks.begin()); it != free_blocks.end(); ++it) {
      auto block = *it;
      if (block.is_original()) {
        if (!synchronized) {
          RMM_CUDA_TRY(cudaStreamSynchronize(stream));
          synchronized = true;
        }
        upstream_mr_->deallocate(block.pointer(), block.size(), cudaStreamLegacy);
        free_blocks.erase(it--);
      }
    }
  }

  bool deallocate_in_arena(
    arena& arena, void* p, size_t bytes, cudaStream_t stream, bool shrink = true)
  {
    lock_guard lock(arena.mtx);
    const auto b = free_block(p, bytes, arena.allocated_blocks);
    if (b.is_valid() && arena.free_blocks.insert(b) && shrink) {
      shrink_arena(arena.free_blocks, stream);
    }
    return b.is_valid();
  }

  void deallocate_across_arena(void* p, size_t bytes, cudaStream_t stream)
  {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    for (auto& kv : arenas_) {
      if (deallocate_in_arena(kv.second, p, bytes, stream, /*shrink=*/false)) return;
    }
  }

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size in bytes of the allocation
   * @param stream The stream to associate this allocation with
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    if (bytes <= 0) return nullptr;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (stream == cudaStreamDefault || stream == cudaStreamPerThread) {
      bytes = rmm::detail::align_up(bytes, allocation_alignment);
      RMM_EXPECTS(
        bytes <= get_maximum_allocation_size(), rmm::bad_alloc, "Maximum allocation size exceeded");

      auto& this_arena = get_arena();
      lock_guard lock(this_arena.mtx);
      auto const b = get_block(this_arena.free_blocks, bytes);
      auto split   = allocate_from_block(b, bytes, this_arena.allocated_blocks);
      if (split.remainder.is_valid()) this_arena.free_blocks.insert(split.remainder);
      return split.allocated_pointer;
    } else {
      return upstream_mr_->allocate(bytes, stream);
    }
#else
    return upstream_mr_->allocate(bytes, stream);
#endif
  }
  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    if (p == nullptr || bytes <= 0) return;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (stream == cudaStreamDefault || stream == cudaStreamPerThread) {
      bytes = rmm::detail::align_up(bytes, allocation_alignment);
      if (deallocate_in_arena(get_arena(), p, bytes, stream)) {
        return;
      } else {
        deallocate_across_arena(p, bytes, stream);
      }
    } else {
      return upstream_mr_->deallocate(p, bytes, stream);
    }
#else
    return upstream_mr_->deallocate(p, bytes, stream);
#endif
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair containing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    return std::make_pair(0, 0);
  }

  /**
   * @brief Clear arenas.
   *
   * Note: only called by destructor.
   */
  void release()
  {
    lock_guard lock(mtx_);

    for (auto& kv : arenas_) {
      auto& arena = kv.second;
      lock_guard arena_lock(arena.mtx);
      arena.free_blocks.clear();
      arena.allocated_blocks.clear();
    }
  }

  arena& get_arena()
  {
    auto thread_id = std::this_thread::get_id();
    auto it        = arenas_.find(thread_id);
    if (it != arenas_.end()) {
      return it->second;
    } else {
      lock_guard lock(mtx_);
      return arenas_[thread_id];
    }
  }

  Upstream* upstream_mr_;  // The upstream memory_resource from which to allocate blocks.
  std::unordered_map<std::thread::id, arena> arenas_;
  mutable std::mutex mtx_;
};

}  // namespace mr
}  // namespace rmm
