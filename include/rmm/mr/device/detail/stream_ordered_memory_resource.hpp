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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <map>
#include <mutex>
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

 protected:
  using free_list  = FreeListType;
  using lock_guard = std::lock_guard<std::mutex>;

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

    lock_guard lock(mtx_);

    stream_event_pair stream_event = get_event(stream);
    bytes                          = rmm::detail::align_up(bytes, allocation_alignment);
    // block const b                  = available_larger_block(bytes, stream_event);
    // auto p                         = allocate_from_block(b, bytes, stream_event);
    void* p = nullptr;
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
    // free_block(p, bytes, stream);
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
  template <typename ParentType>
  struct default_stream_event {
    default_stream_event(ParentType* parent) : parent(parent)
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
    ParentType* parent;
  };

  using default_stream_event_type =
    default_stream_event<stream_ordered_suballocator_memory_resource<FreeListType>>;
#endif

  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

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
      static thread_local default_stream_event_type e{this};
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
      stream_free_blocks_[get_event(cudaStreamLegacy)].insert(std::move(blocks));
      stream_free_blocks_.erase(free_list_iter);

      auto result = cudaEventSynchronize(stream_event.event);
      assert(cudaSuccess == result);
    }
    auto result = cudaEventDestroy(stream_event.event);
    assert(cudaSuccess == result);
  }

  // map of stream_event_pair --> free_list
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  // bidirectional mapping between non-default streams and events
  std::unordered_map<cudaStream_t, stream_event_pair> stream_events_;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  // references to per-thread events to avoid use-after-free when threads exit after MR is deleted
  std::list<std::reference_wrapper<default_stream_event_type>> ptds_events_;
#endif

  std::mutex mutable mtx_;  // mutex for thread-safe access
};

}  // namespace detail
}  // namespace mr
}  // namespace rmm
