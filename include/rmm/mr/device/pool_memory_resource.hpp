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
class pool_memory_resource final : public device_memory_resource {
 public:
  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  // TODO use rmm-level def of this.
  static constexpr size_t allocation_alignment = 256;

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial
   * device memory pool using `upstream_mr`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Size, in bytes, of the initial pool. When
   * zero, an implementation-defined pool size is used.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to.
   */
  explicit pool_memory_resource(Upstream* upstream_mr,
                                std::size_t initial_pool_size = default_initial_size,
                                std::size_t maximum_pool_size = default_maximum_size)
    : upstream_mr_{upstream_mr}, maximum_pool_size_{maximum_pool_size}
  {
    RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");

    cudaDeviceProp props;
    int device{0};
    RMM_CUDA_TRY(cudaGetDevice(&device));
    RMM_CUDA_TRY(cudaGetDeviceProperties(&props, device));

    if (initial_pool_size == default_initial_size) { initial_pool_size = props.totalGlobalMem / 2; }

    initial_pool_size = rmm::detail::align_up(initial_pool_size, allocation_alignment);

    if (maximum_pool_size == default_maximum_size) maximum_pool_size_ = props.totalGlobalMem;

    // Allocate initial block and insert into free list for the legacy default stream
    stream_free_blocks_[get_event(cudaStreamLegacy)].insert(
      block_from_upstream(initial_pool_size, 0));
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource()
  {
    // foo
    release();
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    for (auto& event : ptds_events_)
      event.get().parent = nullptr;
#endif
  }

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

 private:
  using id_type    = uint32_t;
  using block      = rmm::mr::detail::block;
  using free_list  = rmm::mr::detail::free_list<>;
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief A structure pairing a CUDA stream and an associated event for the stream.
   *
   */
  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E, if
   * `CUDA_API_PER_THREAD_DEFAULT_STREAM` is defined, `stream_event.stream` will be made to wait on
   * event E. Otherwise, the stream associated with free list F will be synchronized. In either
   * case all other blocks in free list F will be moved to the free list associated with
   * `stream_event.stream`. This results in coalescing with other blocks in that free list,
   * hopefully reducing fragmentation.
   *
   * @param size The requested size of the allocation.
   * @param stream_event The stream and associated event on which the allocation is being requested.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block get_block_from_other_stream(size_t size, stream_event_pair stream_event)
  {
    // nothing in this stream's free list, look for one on another stream
    for (auto s = stream_free_blocks_.begin(); s != stream_free_blocks_.end(); ++s) {
      auto blocks_event = s->first;
      if (blocks_event.event != stream_event.event) {
        auto blocks = s->second;

        block const b = blocks.best_fit(size);  // get the best fit block

        if (b.is_valid()) {
          // Since we found a block associated with a different stream, we have to insert a wait on
          // the stream's associated event into the allocating stream.
          // TODO: could eliminate this ifdef and have the same behavior for PTDS and non-PTDS
          // But the cudaEventRecord() on every free_block reduces performance significantly
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
          RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, blocks_event.event, 0));
#else
          RMM_CUDA_TRY(cudaStreamSynchronize(blocks_event.stream));
#endif
          // Move all the blocks to the requesting stream, since it has waited on them
          stream_free_blocks_[stream_event].insert(blocks.begin(), blocks.end());
          stream_free_blocks_.erase(s);

          return b;
        }
      }
    }
    return block{};
  }

  /**
   * @brief Find an available block in the pool of at least `size` bytes, for use on `stream`.
   *
   * Attempts to find a free block that was last used on `stream` to avoid synchronization. If
   * none is available, it finds a block last used on another stream. In this case, the stream
   * associated with the found block is synchronized to ensure all asynchronous work on the memory
   * is finished before it is used on `stream`.
   *
   * @throw `std::bad_alloc` if the requested allocation could not be fulfilled.
   *
   * @param size The size of the requested allocation, in bytes.
   * @param stream_event The stream and associated event on which the allocation is being requested.
   * @return block A block with non-null pointer and size >= `size`.
   */
  block available_larger_block(size_t size, stream_event_pair stream_event)
  {
    // Try to find a larger block in free list for the same stream (no sync required)
    auto iter = stream_free_blocks_.find(stream_event);
    if (iter != stream_free_blocks_.end()) {
      block b = iter->second.best_fit(size);
      if (b.is_valid()) return b;
    }

    block b = get_block_from_other_stream(size, stream_event);
    if (b.is_valid()) return b;

    // no larger blocks available on other streams, so grow the pool and create a block
    size_t grow_size = size_to_grow(size);
    RMM_EXPECTS(grow_size > 0, rmm::bad_alloc, "Maximum pool size exceeded");
    return block_from_upstream(grow_size, stream_event.stream);
  }

  /**
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return void* The pointer to the allocated memory.
   */
  void* allocate_from_block(block const& b, size_t size, stream_event_pair stream_event)
  {
    block const alloc{b.pointer(), size, b.is_head()};

    if (b.size() > size) {
      block rest{b.pointer() + size, b.size() - size, false};
      stream_free_blocks_[stream_event].insert(rest);
    }

    allocated_blocks_.insert(alloc);
    return reinterpret_cast<void*>(alloc.pointer());
  }

  /**
   * @brief Frees the block associated with pointer `p`, returning it to the pool.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @param stream The stream on which the memory was last used.
   */
  void free_block(void* p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    stream_event_pair stream_event = get_event(stream);

    auto const i = allocated_blocks_.find(static_cast<char*>(p));
    assert(i != allocated_blocks_.end());
    assert(i->size() == rmm::detail::align_up(size, allocation_alignment));

    // TODO: cudaEventRecord has significant overhead on deallocations, however it could mean less
    // synchronization So we need to test in real non-PTDS applications that have multiple streams
    // whether or not the overhead is worth it
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, stream));
#endif

    stream_free_blocks_[stream_event].insert(*i);
    allocated_blocks_.erase(i);
  }

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   *
   * Current strategy is to try to grow the pool by half the difference between
   * the configured maximum pool size and the current pool size.
   *
   * @param size The size of the minimum allocation immediately needed
   * @return size_t The computed size to grow the pool.
   */
  size_t size_to_grow(size_t size) const
  {
    auto const remaining =
      rmm::detail::align_up(maximum_pool_size_ - pool_size(), allocation_alignment);
    auto const aligned_size = rmm::detail::align_up(size, allocation_alignment);
    if (aligned_size <= remaining / 2) {
      return remaining / 2;
    } else if (aligned_size <= remaining) {
      return remaining;
    } else {
      return 0;
    }
  };

  /**
   * @brief Allocates memory of `size` bytes using the upstream memory_resource, on `stream`.
   *
   * @param size The size of the requested allocation.
   * @param stream The stream on which the requested allocation will be used.
   * @return block A block of at least `size` bytes.
   */
  block block_from_upstream(size_t size, cudaStream_t stream)
  {
    void* p = upstream_mr_->allocate(size, stream);
    block b{reinterpret_cast<char*>(p), size, true};
    upstream_blocks_.emplace_back(b);
    current_pool_size_ += b.size();
    return b;
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
    lock_guard lock(mtx_);

    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.pointer(), b.size());
    upstream_blocks_.clear();
    allocated_blocks_.clear();

    for (auto s_e : stream_events_)
      destroy_event(s_e.second);
    stream_events_.clear();
    stream_free_blocks_.clear();

    current_pool_size_ = 0;
  }

#ifndef NDEBUG
  /**
   * @brief Print debugging information about all blocks in the pool.
   *
   */
  void print()
  {
    lock_guard lock(mtx_);

    std::size_t free, total;
    std::tie(free, total) = upstream_mr_->get_mem_info(0);
    std::cout << "GPU free memory: " << free << "total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto h : upstream_blocks_) {
      h.print();
      upstream_total += h.size();
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
    for (auto b : allocated_blocks_) {
      b.print();
    }

    std::cout << "sync free blocks: ";
    for (auto s : stream_free_blocks_) {
      std::cout << "stream: " << s.first.stream << " event: " << s.first.event << " ";
      s.second.print();
    }
    std::cout << "\n";
  }
#endif  // DEBUG

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @param The stream to associate this allocation with
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    if (bytes <= 0) return nullptr;

    lock_guard lock(mtx_);

    stream_event_pair stream_event = get_event(stream);
    bytes                          = rmm::detail::align_up(bytes, allocation_alignment);
    block const b                  = available_larger_block(bytes, stream_event);
    auto p                         = allocate_from_block(b, bytes, stream_event);
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    lock_guard lock(mtx_);
    free_block(p, bytes, stream);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  /**
   * @brief RAII wrapper for a CUDA event for a per-thread default stream
   *
   * These objects take care of creating and freeing an event associated with a per-thread default
   * stream. They are needed because the event needs to exist in thread_local memory, so it must
   * be cleaned up when the thread exits. They maintain a pointer to the parent
   * (pool_memory_resource) that created them, because when a thread exits, if the parent still
   * exists, they must tell the parent to merge the free list associated with the event. Also, the
   * parent maintains a list of references to the created cuda_event objects so that if any remain
   * when the parent is destroyed, it can set their parent pointers to nullptr to we don't have a
   * use-after-free race. Note: all of this is a workaround for the fact that there is no way
   * currently to get a unique handle to a CUDA per-thread default stream. :(
   */
  struct default_stream_event {
    default_stream_event(pool_memory_resource<Upstream>* parent) : parent(parent)
    {
      auto result = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      assert(cudaSuccess == result);
      if (parent) parent->ptds_events_.push_back(*this);
    }
    ~default_stream_event()
    {
      if (parent) {
        lock_guard lock(parent->mtx_);
        parent->destroy_event(stream_event_pair{cudaStreamDefault, event});
      }
    }

    cudaEvent_t event;
    pool_memory_resource<Upstream>* parent;
  };
#endif

  /**
   * @brief get a unique CUDA event (possibly new) associated with `stream`
   *
   * The event is created on the first call, and it is not recorded. If compiled for per-thread
   * default stream and `stream` is the default stream, the event is created in thread local memory
   * and is unique per CPU thread.
   *
   * @param stream The stream for which to get an event.
   * @return The stream_event for `stream`.
   */
  stream_event_pair get_event(cudaStream_t stream)
  {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (cudaStreamDefault == stream || cudaStreamPerThread == stream) {
      static thread_local default_stream_event e{this};
      return stream_event_pair{stream, e.event};
    }
#else
    // We use cudaStreamLegacy as the event map key for the default stream for consistency between
    // PTDS and non-PTDS mode. In PTDS mode, the cudaStreamLegacy map key will only exist if the
    // user explicitly passes it, so it is used as the default location for the free list
    // at construction, and for merging free lists when a thread exits (see destroy_event()).
    // For consistency, the same key is used for null stream free lists in non-PTDS mode.
    if (cudaStreamDefault == stream) { stream = cudaStreamLegacy; }
#endif

    auto iter = stream_events_.find(stream);
    if (iter == stream_events_.end()) {
      stream_event_pair stream_event{stream};
      auto result = cudaEventCreateWithFlags(&stream_event.event, cudaEventDisableTiming);
      assert(cudaSuccess == result);
      stream_events_[stream] = stream_event;
      return stream_event;
    } else {
      return iter->second;
    }
  }

  /**
   * @brief Destroy the specified CUDA event and move all free blocks for the associated stream
   * to the default stream free list.
   *
   * @param event The event to destroy.
   */
  void destroy_event(stream_event_pair stream_event)
  {
    // If we are destroying an event with associated free list, we need to synchronize that event
    // and then merge its free list into the (legacy) default stream's list
    auto free_list_iter = stream_free_blocks_.find(stream_event);
    if (free_list_iter != stream_free_blocks_.end()) {
      auto blocks = free_list_iter->second;
      stream_free_blocks_[get_event(cudaStreamLegacy)].insert(blocks.begin(), blocks.end());
      stream_free_blocks_.erase(free_list_iter);

      auto result = cudaEventSynchronize(stream_event.event);
      assert(cudaSuccess == result);
    }
    auto result = cudaEventDestroy(stream_event.event);
    assert(cudaSuccess == result);
  }

  size_t maximum_pool_size_;
  size_t current_pool_size_{0};

  Upstream* upstream_mr_;  // The "heap" to allocate the pool from

  // map of [cudaEvent_t, free_list] pairs
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  std::set<block, rmm::mr::detail::compare_blocks<block>> allocated_blocks_;

  // blocks allocated from upstream: so they can be easily freed
  std::vector<block> upstream_blocks_;

  // bidirectional mapping between non-default streams and events
  std::unordered_map<cudaStream_t, stream_event_pair> stream_events_;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  // references to per-thread events to avoid use-after-free when threads exit after MR is deleted
  std::list<std::reference_wrapper<default_stream_event>> ptds_events_;
#endif

  std::mutex mutable mtx_;  // mutex for thread-safe access
};

}  // namespace mr
}  // namespace rmm
