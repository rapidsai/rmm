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

#include <limits>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief
 */
template <typename FreeListType>
class stream_ordered_suballocator_memory_resource : public device_memory_resource {
 public:
  // TODO use rmm-level def of this.
  static constexpr size_t allocation_alignment = 256;

  ~stream_ordered_suballocator_memory_resource() { release(); }

  stream_ordered_suballocator_memory_resource() = default;
  stream_ordered_suballocator_memory_resource(stream_ordered_suballocator_memory_resource const&) =
    delete;
  stream_ordered_suballocator_memory_resource(stream_ordered_suballocator_memory_resource&&) =
    delete;
  stream_ordered_suballocator_memory_resource& operator =(
    stream_ordered_suballocator_memory_resource const&) = delete;
  stream_ordered_suballocator_memory_resource& operator =(
    stream_ordered_suballocator_memory_resource&&) = delete;

 protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using lock_guard = std::lock_guard<std::mutex>;

  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

  /**
   * @brief Get the maximum size of a single allocation supported by this suballocator memory
   * resource
   *
   * Default implementation is the maximum `size_t` value, but fixed-size allocators will have a
   * lower limit. Override this function in derived classes as necessary.
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  virtual size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  /**
   * @brief Allocate space (typically from upstream) to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * This function returns a block because in some suballocators, a single block is allocated
   * from upstream and returned. In other suballocators, many blocks are created from upstream. In
   * the latter case, the function returns one block and inserts all the rest into the free list
   * `blocks`.
   *
   * @param size The minimum size block to return
   * @param blocks The free list into which to optionally insert new blocks
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  virtual block_type expand_pool(size_t size, free_list& blocks, cudaStream_t stream) = 0;

  /**
   * @brief Split block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned as the second element in the output pair.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  virtual std::pair<void*, block_type> allocate_from_block(block_type const& b,
                                                           size_t size,
                                                           stream_event_pair stream_event) = 0;

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @param stream_event The stream-event pair for the stream on which the memory was last used.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  virtual block_type free_block(void* p, size_t size) noexcept = 0;

  /**
   * @brief Returns the block `b` (last used on stream `stream_event`) to the pool.
   *
   * @param b The block to insert into the pool.
   * @param stream The stream on which the memory was last used.
   */
  void insert_block(block_type const& b, cudaStream_t stream)
  {
    stream_free_blocks_[get_event(stream)].insert(b);
  }

  void insert_blocks(free_list&& blocks, cudaStream_t stream)
  {
    stream_free_blocks_[get_event(stream)].insert(std::move(blocks));
  }

  /**
   * @brief Get the mutex object
   *
   * @return std::mutex
   */
  std::mutex& get_mutex() { return mtx_; }

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
  virtual void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    if (bytes <= 0) return nullptr;

    lock_guard lock(mtx_);

    auto stream_event = get_event(stream);
    bytes             = rmm::detail::align_up(bytes, allocation_alignment);
    RMM_EXPECTS(
      bytes <= get_maximum_allocation_size(), rmm::bad_alloc, "Maximum allocation size exceeded");
    auto const b       = get_block(bytes, stream_event);
    auto ptr_remainder = allocate_from_block(b, bytes, stream_event);
    if (is_valid(ptr_remainder.second))
      stream_free_blocks_[stream_event].insert(ptr_remainder.second);
    return ptr_remainder.first;
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   */
  virtual void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    lock_guard lock(mtx_);
    auto stream_event = get_event(stream);
    bytes             = rmm::detail::align_up(bytes, allocation_alignment);
    auto const b      = free_block(p, bytes);

    // TODO: cudaEventRecord has significant overhead on deallocations, however it could mean less
    // synchronization. So we need to test in real non-PTDS applications that have multiple streams
    // whether or not the overhead is worth it
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, stream));
#endif

    stream_free_blocks_[stream_event].insert(b);
  }

 private:
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
  template <typename ParentType>
  struct default_stream_event {
    default_stream_event(ParentType* parent, cudaEvent_t event) : parent(parent), event(event)
    {
      parent->ptds_events_.insert(this);
    }
    ~default_stream_event()
    {
      if (parent) {
        lock_guard lock(parent->get_mutex());
        parent->ptds_events_.erase(this);
        parent->destroy_event(stream_event_pair{cudaStreamDefault, event});
      }
    }

    cudaEvent_t event;
    ParentType* parent;
  };

  using default_stream_event_type =
    default_stream_event<stream_ordered_suballocator_memory_resource<FreeListType>>;
#endif

  /**
   * @brief get a unique CUDA event (possibly new) associated with `stream`
   *
   * The event is created on the first call, and it is not recorded. If compiled for per-thread
   * default stream and `stream` is the default stream, the event is created in thread local
   * memory and is unique per CPU thread.
   *
   * @param stream The stream for which to get an event.
   * @return The stream_event for `stream`.
   */
  stream_event_pair get_event(cudaStream_t stream)
  {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (cudaStreamDefault == stream || cudaStreamPerThread == stream) {
      // Because multiple instances of stream_ordered_memory_resource can be called from the same
      // thread it is not enough to store a single thread-local default_stream_event. We have to
      // store one per MR instance, which we do using a map of instance to default_stream_event.
      // This ensures that the events and their free-lists can be cleaned up at thread exit.
      static thread_local std::unordered_map<void*, default_stream_event_type> default_events_tls;
      auto thread_event = default_events_tls.find(this);
      return (thread_event != default_events_tls.end())
               ? stream_event_pair{stream, thread_event->second.event}
               : [&]() {
                   cudaEvent_t event;
                   RMM_ASSERT_CUDA_SUCCESS(
                     cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                   default_events_tls.emplace(std::piecewise_construct,
                                              std::forward_as_tuple(this),
                                              std::forward_as_tuple(this, event));
                   return stream_event_pair{stream, event};
                 }();
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
      RMM_ASSERT_CUDA_SUCCES(cudaEventCreateWithFlags(&stream_event.event, cudaEventDisableTiming));
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
      stream_free_blocks_[get_event(cudaStreamLegacy)].insert(std::move(blocks));
      stream_free_blocks_.erase(free_list_iter);

      RMM_ASSERT_CUDA_SUCCESS(cudaEventSynchronize(stream_event.event));
    }
    RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(stream_event.event));
  }

  /**
   * @brief Get an avaible memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(size_t size, stream_event_pair stream_event)
  {
    // Try to find a satisfactory block in free list for the same stream (no sync required)
    auto iter = stream_free_blocks_.find(stream_event);
    if (iter != stream_free_blocks_.end()) {
      block_type b = iter->second.get_block(size);
      if (is_valid(b)) return b;
    }

    // Otherwise try to find a block associated with another stream
    block_type b = get_block_from_other_stream(size, stream_event);
    if (is_valid(b)) return b;

    // no larger blocks available on other streams, so grow the pool and create a block

    // avoid searching for this stream's list again
    free_list& blocks =
      (iter != stream_free_blocks_.end()) ? iter->second : stream_free_blocks_[stream_event];

    return expand_pool(size, blocks, stream_event.stream);
  }

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E, if
   * `CUDA_API_PER_THREAD_DEFAULT_STREAM` is defined, `stream_event.stream` will be made to wait
   * on event E. Otherwise, the stream associated with free list F will be synchronized. In either
   * case all other blocks in free list F will be moved to the free list associated with
   * `stream_event.stream`. This results in coalescing with other blocks in that free list,
   * hopefully reducing fragmentation.
   *
   * @param size The requested size of the allocation.
   * @param stream_event The stream and associated event on which the allocation is being
   * requested.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block_type get_block_from_other_stream(size_t size, stream_event_pair stream_event)
  {
    // nothing in this stream's free list, look for one on another stream
    for (auto s = stream_free_blocks_.begin(); s != stream_free_blocks_.end(); ++s) {
      auto blocks_event = s->first;
      if (blocks_event.event != stream_event.event) {
        auto blocks = s->second;

        block_type const b = blocks.get_block(size);  // get the best fit block

        if (is_valid(b)) {
          // Since we found a block associated with a different stream, we have to insert a wait
          // on the stream's associated event into the allocating stream.
          // TODO: could eliminate this ifdef and have the same behavior for PTDS and non-PTDS
          // But the cudaEventRecord() on every free_block reduces performance significantly
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
          RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, blocks_event.event, 0));

#else
          RMM_CUDA_TRY(cudaStreamSynchronize(blocks_event.stream));
#endif
          // Move all the blocks to the requesting stream, since it has waited on them
          stream_free_blocks_[stream_event].insert(std::move(blocks));
          stream_free_blocks_.erase(s);

          return b;
        }
      }
    }
    return block_type{};
  }

  /**
   * @brief Clear free lists and events
   *
   * Note: only called by destructor.
   */
  void release()
  {
    lock_guard lock(mtx_);

    for (auto s_e : stream_events_)
      destroy_event(s_e.second);
    stream_events_.clear();
    stream_free_blocks_.clear();

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    for (auto& event : ptds_events_)
      reinterpret_cast<default_stream_event_type*>(event)->parent = nullptr;
#endif
  }

  // map of stream_event_pair --> free_list
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  // bidirectional mapping between non-default streams and events
  std::unordered_map<cudaStream_t, stream_event_pair> stream_events_;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  // pointers to per-thread events to avoid use-after-free when threads exit after MR is deleted
  std::set<void*> ptds_events_;
#endif

  std::mutex mtx_;  // mutex for thread-safe access
};                  // namespace detail

}  // namespace detail
}  // namespace mr
}  // namespace rmm
