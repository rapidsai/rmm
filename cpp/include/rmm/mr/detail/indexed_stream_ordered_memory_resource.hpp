/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/logger.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <map>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>
#ifdef RMM_DEBUG_PRINT
#include <iostream>
#endif

RMM_NAMESPACE_BEGIN
namespace mr::detail {

/**
 * @brief Fault-injection hooks for selective recovery tests.
 *
 * Hooks are thread-local so concurrent tests cannot affect allocator calls on other threads.
 * Metadata checkpoints precede every staging allocation required for the no-throw commit phase.
 */
#ifdef RMM_INDEXED_RECOVERY_DISABLE_TEST_HOOKS
struct indexed_recovery_test_hooks {
  static cudaError_t wait(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
  { return cudaStreamWaitEvent(stream, event, flags); }

  static cudaError_t record(cudaEvent_t event, cudaStream_t stream)
  { return cudaEventRecord(event, stream); }

  static void metadata_checkpoint() noexcept {}
};
#else
struct indexed_recovery_test_hooks {
  using wait_function   = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
  using record_function = cudaError_t (*)(cudaEvent_t, cudaStream_t);

  static cudaError_t default_wait(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
  { return cudaStreamWaitEvent(stream, event, flags); }

  static cudaError_t default_record(cudaEvent_t event, cudaStream_t stream)
  { return cudaEventRecord(event, stream); }

  static void metadata_checkpoint()
  {
    if (metadata_fail_after == 0) { throw std::bad_alloc{}; }
    if (metadata_fail_after > 0) { --metadata_fail_after; }
  }

  static void reset() noexcept
  {
    wait                = default_wait;
    record              = default_record;
    metadata_fail_after = -1;
  }

  inline static thread_local wait_function wait{default_wait};
  inline static thread_local record_function record{default_record};
  inline static thread_local int metadata_fail_after{-1};
};
#endif

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/indexed_crtp-helper/
 *
 * Does two things:
 * 1. Makes "indexed_crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template <typename T>
struct indexed_crtp {
  [[nodiscard]] T& underlying() { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief Base class for a stream-ordered memory resource
 *
 * This base class uses CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * to provide static polymorphism to enable defining suballocator resources that maintain separate
 * pools per stream. All of the stream-ordering logic is contained in this class, but the logic
 * to determine how memory pools are managed and the type of allocation is implemented in a derived
 * class and in a free list class.
 *
 * For example, a coalescing pool memory resource uses a coalescing_free_list and maintains data
 * structures for allocated blocks and has functions to allocate and free blocks and to expand the
 * pool.
 *
 * Classes derived from indexed_stream_ordered_memory_resource must implement the following methods,
 * documented separately:
 *
 * 1. `std::size_t get_maximum_allocation_size() const`
 * 2. `block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)`
 * 3. `split_block allocate_from_block(block_type const& b, std::size_t size)`
 * 4. `prepared_allocation_tracking prepare_allocation_tracking(block_type const& b)`
 * 5. `void commit_allocation_tracking(prepared_allocation_tracking&& prepared) noexcept`
 * 6. `block_type free_block(void* ptr, std::size_t size) noexcept`
 */
template <typename PoolResource, typename FreeListType>
class indexed_stream_ordered_memory_resource : public indexed_crtp<PoolResource> {
 public:
  ~indexed_stream_ordered_memory_resource() { release(); }

  indexed_stream_ordered_memory_resource()                                              = default;
  indexed_stream_ordered_memory_resource(indexed_stream_ordered_memory_resource const&) = delete;
  indexed_stream_ordered_memory_resource(indexed_stream_ordered_memory_resource&&)      = delete;
  indexed_stream_ordered_memory_resource& operator=(indexed_stream_ordered_memory_resource const&) =
    delete;
  indexed_stream_ordered_memory_resource& operator=(indexed_stream_ordered_memory_resource&&) =
    delete;

  /**
   * @brief Allocates memory of size at least `bytes` bytes.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param stream The stream in which to order this allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment Unused; alignment is always at least `CUDA_ALLOCATION_ALIGNMENT`
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t /*alignment*/)
  {
    auto const strm = cuda_stream_view{stream};

    RMM_LOG_TRACE("[A][stream %s][%zuB]", rmm::detail::format_stream(strm), bytes);

    if (bytes == 0) { return nullptr; }

    lock_guard lock(mtx_);

    auto stream_event = get_event(strm);

    bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    RMM_EXPECTS(bytes <= this->underlying().get_maximum_allocation_size(),
                std::string("Maximum allocation size exceeded (failed to allocate ") +
                  rmm::detail::format_bytes(bytes) + ")",
                rmm::out_of_memory);
    auto const block = this->underlying().get_block(bytes, stream_event);

    RMM_LOG_TRACE("[A][stream %s][%zuB][%p]",
                  rmm::detail::format_stream(stream_event.stream),
                  bytes,
                  block.pointer());

    log_summary_trace();

    return block.pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Unused
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t /*alignment*/) noexcept
  {
    auto const strm = cuda_stream_view{stream};

    RMM_LOG_TRACE("[D][stream %s][%zuB][%p]", rmm::detail::format_stream(strm), bytes, ptr);

    if (bytes == 0 || ptr == nullptr) { return; }

    lock_guard lock(mtx_);
    auto stream_event = get_event(strm);

    // Register a new owner before removing allocation tracking or publishing the free block.
    auto& blocks = get_or_create_owner_blocks(stream_event);

    bytes            = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto const block = this->underlying().free_block(ptr, bytes);

    // TODO: cudaEventRecord has significant overhead on deallocations. For the non-PTDS case
    // we may be able to delay recording the event in some situations. But using events rather
    // than streams allows stealing from deleted streams.
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, strm.value()));

    auto const inserted_size = blocks.insert(block);
    total_free_bytes_ += block.size();
    invalidate_global_failure_cache();
    update_stream_maximum_after_insert(stream_event, blocks, inserted_size);

    log_summary_trace();
  }

  /**
   * @brief Allocates memory of size at least `bytes` synchronously.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size in bytes of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
    void* ptr         = allocate(stream, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr` synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Alignment of the allocation
   */
  void deallocate_sync(
    void* ptr,
    std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
    deallocate(stream, ptr, bytes, alignment);
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
  }

 protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using lock_guard = std::lock_guard<std::mutex>;

  using stream_id_type = unsigned long long;  ///< Stream identifier returned by cudaStreamGetId
  // Derived classes must implement these six methods

  /*
   * @brief Get the maximum size of a single allocation supported by this suballocator memory
   * resource
   *
   * Default implementation is the maximum `std::size_t` value, but fixed-size allocators will have
   * a lower limit. Override this function in derived classes as necessary.
   *
   * @return std::size_t The maximum size of a single allocation supported by this memory resource
   */
  // std::size_t get_maximum_allocation_size() const

  /*
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
  // block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)

  /// Pair representing a block that has been split for allocation
  using split_block = std::pair<block_type, block_type>;

  /*
   * @brief Split block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned as the remainder element in the output
   * `split_block`.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A `split_block` comprising the allocated pointer and any unallocated remainder of the
   * input block.
   */
  // split_block allocate_from_block(block_type const& b, std::size_t size)

  // prepared_allocation_tracking prepare_allocation_tracking(block_type const& block)

  // void commit_allocation_tracking(prepared_allocation_tracking&& prepared) noexcept

  /*
   * @brief Finds, frees and returns the block associated with pointer `ptr`.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `ptr`. The caller is expected to return the block
   * to the pool.
   */
  // block_type free_block(void* ptr, std::size_t size) noexcept

  /**
   * @brief Returns the block `b` (last used on stream `stream_event`) to the pool.
   *
   * @param block The block to insert into the pool.
   * @param stream The stream on which the memory was last used.
   */
  void insert_block(block_type const& block, cuda_stream_view stream)
  {
    auto const stream_event = get_event(stream);
    // The upstream allocation that produced this initial block is only ordered on `stream`.
    // Publish that readiness before another stream can discover the block through the index.
    RMM_CUDA_TRY(cudaEventRecord(stream_event.event, stream_event.stream));
    auto& blocks             = stream_free_blocks_[stream_event];
    auto const inserted_size = blocks.insert(block);
    total_free_bytes_ += block.size();
    invalidate_global_failure_cache();
    update_stream_maximum_after_insert(stream_event, blocks, inserted_size);
  }

#ifdef RMM_DEBUG_PRINT
  void print_free_blocks() const
  {
    std::cout << "stream free blocks: ";
    for (auto& free_blocks : stream_free_blocks_) {
      std::cout << "stream: " << free_blocks.first.stream << " event: " << free_blocks.first.event
                << " ";
      free_blocks.second.print();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
#endif

  /**
   * @brief Get the mutex object
   *
   * @return std::mutex
   */
  std::mutex& get_mutex() { return mtx_; }

  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

  struct stream_event_hash {
    std::size_t operator()(stream_event_pair const& value) const
    { return std::hash<cudaEvent_t>{}(value.event); }
  };

  struct stream_event_equal {
    bool operator()(stream_event_pair const& lhs, stream_event_pair const& rhs) const
    { return lhs.event == rhs.event; }
  };

  /**
   * @brief Reclaim entirely-free upstream blocks from every owner list.
   *
   * The callback identifies and removes reclaimable blocks from one owner, enqueuing any upstream
   * deallocation on `stream`. It returns the number of bytes removed from that owner list.
   *
   * @tparam Reclaimer Callable accepting `(free_list&, cudaEvent_t, stream_event_pair)`.
   * @param stream Stream on which reclaimed upstream blocks are deallocated.
   * @param reclaimer Owner-local reclamation callback.
   */
  template <typename Reclaimer>
  void reclaim_free_upstream_blocks(cuda_stream_view stream, Reclaimer&& reclaimer)
  {
    auto const requester = get_event(stream);
    for (auto& [owner, blocks] : stream_free_blocks_) {
      auto const reclaimed = reclaimer(blocks, owner.event, requester);
      if (reclaimed == 0) { continue; }
      total_free_bytes_ -= reclaimed;
      invalidate_global_failure_cache();
      update_stream_maximum(owner, blocks);
    }
  }

 private:
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
  stream_event_pair get_event(cuda_stream_view stream)
  {
    if (stream.is_per_thread_default()) {
      // Create a thread-local event for each device. These events are
      // deliberately leaked since the destructor needs to call into
      // the CUDA runtime and thread_local destructors (can) run below
      // main: it is undefined behaviour to call into the CUDA
      // runtime below main.
      thread_local std::vector<cudaEvent_t> events_tls(
        static_cast<std::size_t>(rmm::get_num_cuda_devices()));
      auto event = [device_id = this->device_id_]() {
        auto& e = events_tls[static_cast<std::size_t>(device_id.value())];
        if (!e) {
          // These events are deliberately not destructed and therefore live until
          // program exit.
          RMM_ASSERT_CUDA_SUCCESS(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        return e;
      }();
      return stream_event_pair{stream.value(), event};
    }
    // We use cudaStreamLegacy as the event map key for the default stream for consistency between
    // PTDS and non-PTDS mode. In PTDS mode, the cudaStreamLegacy map key will only exist if the
    // user explicitly passes it, so it is used as the default location for the free list
    // at construction. For consistency, the same key is used for null stream free lists in
    // non-PTDS mode.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    auto* const stream_to_store = stream.is_default() ? cudaStreamLegacy : stream.value();
    stream_id_type stream_id{};
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamGetId(stream_to_store, &stream_id));
    auto const iter = stream_events_.find(stream_id);
    return (iter != stream_events_.end()) ? iter->second : [&]() {
      stream_event_pair stream_event{stream_to_store, nullptr};
      RMM_ASSERT_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&stream_event.event, cudaEventDisableTiming));
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      stream_events_[stream_id] = stream_event;
      return stream_event;
    }();
  }

  /**
   * @brief Prepares and commits an allocation from an existing selected free block.
   *
   * All potentially throwing host metadata is prepared before a cross-stream dependency is
   * published. The selected list node and its index nodes are then consumed or replaced by the
   * split remainder without allocating.
   */
  block_type allocate_from_selection(typename free_list::block_selection selection,
                                     std::size_t size,
                                     stream_event_pair owner,
                                     stream_event_pair requester,
                                     free_list& blocks)
  {
    block_type const block = *selection.block;
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    auto prepared_tracking = this->underlying().prepare_allocation_tracking(allocated);
#endif

    if (owner.event != requester.event) {
      RMM_CUDA_TRY(indexed_recovery_test_hooks::wait(requester.stream, owner.event, 0));
    }

    // Everything after dependency publication is allocation-free.
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    this->underlying().commit_allocation_tracking(std::move(prepared_tracking));
#endif
    blocks.commit_block_selection(selection, remainder);
    total_free_bytes_ -= size;
    auto const inserted_size = remainder.is_valid() ? remainder.size() : 0;
    update_stream_maximum_after_allocation(owner, blocks, block.size(), inserted_size);
    return allocated;
  }

  block_type allocate_from_expanded_block(block_type block,
                                          std::size_t size,
                                          stream_event_pair owner,
                                          free_list& blocks)
  {
    total_free_bytes_ += block.size();
    invalidate_global_failure_cache();
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    auto prepared_tracking = this->underlying().prepare_allocation_tracking(allocated);
    this->underlying().commit_allocation_tracking(std::move(prepared_tracking));
#endif
    total_free_bytes_ -= size;
    if (remainder.is_valid()) {
      // Upstream allocation readiness is stream ordered. Publish it before the remainder becomes
      // visible to a different stream through the shared index.
      RMM_CUDA_TRY(cudaEventRecord(owner.event, owner.stream));
      auto const inserted_size = blocks.insert(remainder);
      update_stream_maximum_after_insert(owner, blocks, inserted_size);
    }
    return allocated;
  }

  /**
   * @brief Return an existing owner list or create and pre-register an empty one.
   *
   * Maximum-index registration happens before callers publish CUDA dependencies or mutate
   * allocation/free-list ownership. If registration fails, the empty owner list is removed.
   */
  free_list& get_or_create_owner_blocks(stream_event_pair owner)
  {
    auto const existing = stream_free_blocks_.find(owner);
    if (existing != stream_free_blocks_.end()) { return existing->second; }

    auto const [inserted, was_inserted] = stream_free_blocks_.try_emplace(owner);
    assert(was_inserted);
    try {
      update_stream_maximum(owner, inserted->second);
    } catch (...) {
      stream_free_blocks_.erase(inserted);
      throw;
    }
    return inserted->second;
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(std::size_t size, stream_event_pair stream_event)
  {
    // Preserve the synchronization-free same-stream fast path.
    auto iter = stream_free_blocks_.find(stream_event);
    if (iter != stream_free_blocks_.end()) {
      auto const selection = iter->second.find_block(size);
      if (selection.block != iter->second.end()) {
        return allocate_from_selection(
          selection, size, stream_event, stream_event, iter->second);
      }
    }

    free_list& blocks =
      (iter != stream_free_blocks_.end()) ? iter->second : get_or_create_owner_blocks(stream_event);
    if (iter != stream_free_blocks_.end()) { activate_maximum_index_if_needed(); }

    // The shared maximum index identifies a fitting owner without walking every stream list once
    // enough owners exist to amortize maintaining it.
    {
      block_type const block = get_block_from_other_stream(size, stream_event);
      if (block.is_valid()) { return block; }
    }

    // No individual block can satisfy the request. Find one contiguous cross-owner run and
    // publish dependencies only for the owners whose blocks are consumed.
    {
      block_type const block = recover_selectively(size, stream_event, blocks);
      if (block.is_valid()) { return block; }
    }

    log_summary_trace();

    // No large enough block is available after recovery, so grow the shared pool.
    block_type const block =
      this->underlying().expand_pool(size, blocks, cuda_stream_view{stream_event.stream});

    return allocate_from_expanded_block(block, size, stream_event, blocks);
  }

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E,
   * `stream_event.stream` will be made to wait on event E.
   *
   * @param size The requested size of the allocation.
   * @param stream_event The stream and associated event on which the allocation is being
   * requested.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block_type get_block_from_other_stream(std::size_t size, stream_event_pair stream_event)
  {
    if (!maximum_index_active_) {
      for (auto& [owner, owner_blocks] : stream_free_blocks_) {
        if (owner.event == stream_event.event) { continue; }

        auto const selection = owner_blocks.find_block(size);
        if (selection.block == owner_blocks.end()) { continue; }

        auto const allocated =
          allocate_from_selection(selection, size, owner, stream_event, owner_blocks);
        RMM_LOG_DEBUG("[A][Stream %s][%zuB][Taken from stream %s]",
                      rmm::detail::format_stream(stream_event.stream),
                      size,
                      rmm::detail::format_stream(owner.stream));
        return allocated;
      }
      return {};
    }

    auto candidate = streams_by_maximum_.lower_bound(size);
    while (candidate != streams_by_maximum_.end() &&
           candidate->second.event == stream_event.event) {
      ++candidate;
    }
    if (candidate == streams_by_maximum_.end()) { return {}; }

    auto const owner = candidate->second;
    auto owner_iter  = stream_free_blocks_.find(owner);
    assert(owner_iter != stream_free_blocks_.end());

    auto& owner_blocks      = owner_iter->second;
    auto const selection = owner_blocks.find_block(size);
    assert(selection.block != owner_blocks.end());

    // The allocated block moves to the requester, but an untouched split remainder remains owned
    // by the donor event. This avoids relabelling the remainder.
    auto const allocated =
      allocate_from_selection(selection, size, owner, stream_event, owner_blocks);
    RMM_LOG_DEBUG("[A][Stream %s][%zuB][Taken from stream %s]",
                  rmm::detail::format_stream(stream_event.stream),
                  size,
                  rmm::detail::format_stream(owner.stream));
    return allocated;
  }

  using stream_blocks_map     = std::map<stream_event_pair, free_list>;
  using maximum_index        = std::multimap<std::size_t, stream_event_pair>;
  using maximum_lookup_index = std::unordered_map<stream_event_pair,
                                                  typename maximum_index::iterator,
                                                  stream_event_hash,
                                                  stream_event_equal>;

  void invalidate_global_failure_cache() noexcept { global_failed_lookup_ = false; }

  void record_global_failure(std::size_t size) noexcept
  {
    smallest_global_failed_request_ =
      global_failed_lookup_ ? std::min(smallest_global_failed_request_, size) : size;
    global_failed_lookup_ = true;
  }

  struct recovery_entry {
    typename stream_blocks_map::iterator owner;
    typename free_list::iterator block;
  };

  /**
   * @brief Coalesce only the contiguous cross-owner run needed for this allocation.
   *
   * Recovery has a planning phase that may throw, a dependency-publication phase, and a no-throw
   * commit phase. No free-list ownership changes before all metadata and tracking nodes have been
   * prepared and the selected donor events have been made visible through the requester event.
   */
  block_type recover_selectively(std::size_t size,
                                 stream_event_pair stream_event,
                                 free_list& requester_blocks)
  {
    if (global_failed_lookup_ && size >= smallest_global_failed_request_) { return {}; }
    if (total_free_bytes_ < size) {
      record_global_failure(size);
      return {};
    }

    auto const heap_compare = [](recovery_entry const& lhs, recovery_entry const& rhs) {
      return std::less<char*>{}(rhs.block->pointer(), lhs.block->pointer());
    };
    auto make_owner_heap = [&]() {
      indexed_recovery_test_hooks::metadata_checkpoint();
      std::vector<recovery_entry> heap;
      heap.reserve(stream_free_blocks_.size());
      for (auto owner = stream_free_blocks_.begin(); owner != stream_free_blocks_.end(); ++owner) {
        if (!owner->second.is_empty()) {
          heap.push_back(recovery_entry{owner, owner->second.begin()});
        }
      }
      std::make_heap(heap.begin(), heap.end(), heap_compare);
      return heap;
    };
    auto pop_next = [&](std::vector<recovery_entry>& heap) {
      std::pop_heap(heap.begin(), heap.end(), heap_compare);
      auto result = heap.back();
      heap.pop_back();
      auto next = std::next(result.block);
      if (next != result.owner->second.end()) {
        heap.push_back(recovery_entry{result.owner, next});
        std::push_heap(heap.begin(), heap.end(), heap_compare);
      }
      return result;
    };

    auto heap = make_owner_heap();

    // Retain the current contiguous run during the merge. Explicit reserve checkpoints ensure that
    // every possible vector allocation occurs before dependency publication.
    indexed_recovery_test_hooks::metadata_checkpoint();
    std::vector<recovery_entry> selected;
    selected.reserve(heap.size());
    auto retain_selected = [&](recovery_entry entry) {
      if (selected.size() == selected.capacity()) {
        indexed_recovery_test_hooks::metadata_checkpoint();
        selected.reserve(std::max<std::size_t>(1, 2 * selected.capacity()));
      }
      selected.push_back(entry);
    };

    block_type combined{};
    while (!heap.empty()) {
      auto const entry = pop_next(heap);
      if (!combined.is_valid() || !combined.is_contiguous_before(*entry.block)) {
        combined = *entry.block;
        selected.clear();
      } else {
        combined = combined.merge(*entry.block);
      }
      retain_selected(entry);
      if (combined.size() >= size) { break; }
    }

    if (combined.size() < size) {
      record_global_failure(size);
      return {};
    }

    block_type const allocated{combined.pointer(), size, combined.is_head()};
    block_type remainder = combined.size() > size
                             ? block_type{combined.pointer() + size, combined.size() - size, false}
                             : block_type{};

    recovery_entry requester_neighbor{};
    bool coalesce_requester_neighbor{};
    if (remainder.is_valid() && !heap.empty()) {
      auto const next = pop_next(heap);
      if (next.owner->first.event == stream_event.event &&
          remainder.is_contiguous_before(*next.block)) {
        requester_neighbor          = next;
        remainder                   = remainder.merge(*next.block);
        coalesce_requester_neighbor = true;
      }
    }

    struct affected_owner {
      typename stream_blocks_map::iterator owner;
      typename maximum_lookup_index::iterator maximum;
    };

    // Sorting owner records makes donor-event deduplication O(K log K), and the same vector limits
    // post-commit maximum rekeys to lists that recovery actually changes.
    indexed_recovery_test_hooks::metadata_checkpoint();
    std::vector<affected_owner> affected_owners;
    affected_owners.reserve(selected.size() + 1);
    for (auto const& entry : selected) {
      affected_owners.push_back({entry.owner, maximum_by_stream_.end()});
    }
    if (remainder.is_valid()) {
      auto const requester_owner = stream_free_blocks_.find(stream_event);
      assert(requester_owner != stream_free_blocks_.end());
      assert(&requester_owner->second == &requester_blocks);
      affected_owners.push_back({requester_owner, maximum_by_stream_.end()});
    }
    auto const owner_less = [](affected_owner const& lhs, affected_owner const& rhs) {
      return std::less<cudaEvent_t>{}(lhs.owner->first.event, rhs.owner->first.event);
    };
    std::sort(affected_owners.begin(), affected_owners.end(), owner_less);
    affected_owners.erase(
      std::unique(affected_owners.begin(),
                  affected_owners.end(),
                  [](affected_owner const& lhs, affected_owner const& rhs) {
                    return lhs.owner->first.event == rhs.owner->first.event;
                  }),
      affected_owners.end());

    // Resolve shared-index iterators before CUDA publication. Recovery retains every owner record,
    // so active-index commits only rekey existing nodes and cannot allocate.
    if (maximum_index_active_) {
      for (auto& affected : affected_owners) {
        affected.maximum = maximum_by_stream_.find(affected.owner->first);
        assert(affected.maximum != maximum_by_stream_.end());
      }
    }

    free_list remainder_staging;
    typename free_list::prepared_splice prepared_remainder;
    if (remainder.is_valid()) {
      indexed_recovery_test_hooks::metadata_checkpoint();
      remainder_staging.insert(remainder);
      indexed_recovery_test_hooks::metadata_checkpoint();
      requester_blocks.prepare_for_splice(1);
      indexed_recovery_test_hooks::metadata_checkpoint();
      prepared_remainder = requester_blocks.prepare_splice(remainder_staging.begin());
    }

    indexed_recovery_test_hooks::metadata_checkpoint();
#ifdef RMM_POOL_TRACK_ALLOCATIONS
    auto prepared_tracking = this->underlying().prepare_allocation_tracking(allocated);
#endif
    free_list consumed_staging;

    for (auto const& affected : affected_owners) {
      if (affected.owner->first.event != stream_event.event) {
        RMM_CUDA_TRY(
          indexed_recovery_test_hooks::wait(stream_event.stream, affected.owner->first.event, 0));
      }
    }
    RMM_CUDA_TRY(indexed_recovery_test_hooks::record(stream_event.event, stream_event.stream));

#ifdef RMM_POOL_TRACK_ALLOCATIONS
    this->underlying().commit_allocation_tracking(std::move(prepared_tracking));
#endif
    for (auto const& entry : selected) {
      entry.owner->second.extract_exact(entry.block, consumed_staging);
    }
    if (coalesce_requester_neighbor) {
      requester_neighbor.owner->second.extract_exact(requester_neighbor.block, consumed_staging);
    }
    if (remainder.is_valid()) {
      requester_blocks.commit_prepared_splice(
        remainder_staging, remainder_staging.begin(), std::move(prepared_remainder));
    }
    total_free_bytes_ -= size;

    if (maximum_index_active_) {
      for (auto const& affected : affected_owners) {
        rekey_stream_maximum_noexcept(affected.maximum, affected.owner->second);
      }
    }

    return allocated;
  }

  // Fresh-process crossover measurements put index maintenance near break-even at 24 retained
  // stream/event owner records, while preserving the simpler path for the common <=16-owner case.
  // Owner records are intentionally retained and the index remains active after crossing this
  // threshold, avoiding transition churn as streams become empty or are reused.
  static constexpr std::size_t maximum_index_activation_owner_count{24};

  void activate_maximum_index_if_needed()
  {
    if (maximum_index_active_ ||
        stream_free_blocks_.size() <= maximum_index_activation_owner_count) {
      return;
    }

    // Construct both sides in temporaries so allocation failure leaves the inactive state intact.
    maximum_index new_maximums;
    maximum_lookup_index new_lookup;
    new_lookup.reserve(stream_free_blocks_.size());
    for (auto& [owner, blocks] : stream_free_blocks_) {
      auto const entry = new_maximums.emplace(blocks.largest_block_size(), owner);
      new_lookup.emplace(owner, entry);
    }

    streams_by_maximum_.swap(new_maximums);
    maximum_by_stream_.swap(new_lookup);
    maximum_index_active_ = true;
  }

  void erase_stream_maximum(stream_event_pair stream_event)
  {
    auto const existing = maximum_by_stream_.find(stream_event);
    if (existing != maximum_by_stream_.end()) {
      streams_by_maximum_.erase(existing->second);
      maximum_by_stream_.erase(existing);
    }
  }

  void set_stream_maximum(typename maximum_lookup_index::iterator existing,
                          stream_event_pair stream_event,
                          std::size_t maximum)
  {
    if (existing == maximum_by_stream_.end()) {
      auto const entry = streams_by_maximum_.emplace(maximum, stream_event);
      try {
        auto const result = maximum_by_stream_.emplace(stream_event, entry);
        if (!result.second) { streams_by_maximum_.erase(entry); }
      } catch (...) {
        streams_by_maximum_.erase(entry);
        throw;
      }
      return;
    }
    if (existing->second->first == maximum) { return; }

    // Re-key the existing node instead of freeing and allocating a tree node on every maximum
    // change. Zero-sized entries remain in the index and are naturally skipped by lower_bound for
    // every non-zero allocation request.
    auto node        = streams_by_maximum_.extract(existing->second);
    node.key()       = maximum;
    existing->second = streams_by_maximum_.insert(std::move(node));
  }

  /**
   * @brief Rekey one existing maximum-index node without allocation.
   *
   * The lookup iterator is resolved before CUDA dependency publication. Both containers retain
   * their nodes throughout recovery, and node-handle insertion reuses the extracted allocation.
   */
  void rekey_stream_maximum_noexcept(typename maximum_lookup_index::iterator existing,
                                     free_list const& blocks) noexcept
  {
    assert(existing != maximum_by_stream_.end());
    auto const maximum = blocks.largest_block_size();
    if (existing->second->first == maximum) { return; }

    auto node        = streams_by_maximum_.extract(existing->second);
    node.key()       = maximum;
    existing->second = streams_by_maximum_.insert(std::move(node));
  }

  void set_stream_maximum(stream_event_pair stream_event, std::size_t maximum)
  { set_stream_maximum(maximum_by_stream_.find(stream_event), stream_event, maximum); }

  void update_stream_maximum(stream_event_pair stream_event, free_list const& blocks)
  {
    activate_maximum_index_if_needed();
    if (!maximum_index_active_) { return; }
    set_stream_maximum(stream_event, blocks.largest_block_size());
  }

  void update_stream_maximum_after_insert(stream_event_pair stream_event,
                                          free_list const& blocks,
                                          std::size_t inserted_size)
  {
    activate_maximum_index_if_needed();
    if (!maximum_index_active_ || inserted_size == 0) { return; }

    auto const existing = maximum_by_stream_.find(stream_event);
    if (existing != maximum_by_stream_.end() && existing->second->first >= inserted_size) {
      return;
    }

    auto const maximum =
      (existing == maximum_by_stream_.end()) ? blocks.largest_block_size() : inserted_size;
    set_stream_maximum(existing, stream_event, maximum);
  }

  void update_stream_maximum_after_allocation(stream_event_pair stream_event,
                                              free_list const& blocks,
                                              std::size_t allocated_block_size,
                                              std::size_t inserted_size)
  {
    if (!maximum_index_active_) { return; }

    auto const existing = maximum_by_stream_.find(stream_event);
    if (existing == maximum_by_stream_.end()) { return; }

    auto const previous_maximum = existing->second->first;
    if (previous_maximum == allocated_block_size) {
      set_stream_maximum(existing, stream_event, blocks.largest_block_size());
    } else if (inserted_size > previous_maximum) {
      set_stream_maximum(existing, stream_event, inserted_size);
    }
  }

  /**
   * @brief Clear free lists and events
   *
   * Note: only called by destructor.
   */
  void release()
  {
    lock_guard lock(mtx_);

    if (rmm::process_is_exiting()) { return; }

    for (auto s_e : stream_events_) {
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventSynchronize(s_e.second.event));
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventDestroy(s_e.second.event));
    }

    stream_events_.clear();
    maximum_by_stream_.clear();
    streams_by_maximum_.clear();
    maximum_index_active_ = false;
    stream_free_blocks_.clear();
  }

  void log_summary_trace()
  {
#if (RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_TRACE)
    std::size_t num_blocks{0};
    std::size_t max_block{0};
    std::size_t free_mem{0};
    std::for_each(stream_free_blocks_.cbegin(),
                  stream_free_blocks_.cend(),
                  [this, &num_blocks, &max_block, &free_mem](auto const& freelist) {
                    num_blocks += freelist.second.size();
                    auto summary = this->underlying().free_list_summary(freelist.second);
                    max_block    = std::max(summary.first, max_block);
                    free_mem += summary.second;
                  });
    RMM_LOG_TRACE("[Summary][Free lists: %zu][Blocks: %zu][Max Block: %zu][Total Free: %zu]",
                  stream_free_blocks_.size(),
                  num_blocks,
                  max_block,
                  free_mem);
#endif
  }

  // map of stream_event_pair --> free_list
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  // Shared index of each owner's largest block. This makes cross-stream selection logarithmic
  // while each block remains associated with the event that makes it safe to reuse.
  maximum_index streams_by_maximum_;
  maximum_lookup_index maximum_by_stream_;
  bool maximum_index_active_{false};

  // bidirectional mapping between non-default streams and events
  std::unordered_map<stream_id_type, stream_event_pair> stream_events_;

  std::size_t total_free_bytes_{};
  bool global_failed_lookup_{false};
  std::size_t smallest_global_failed_request_{};

  std::mutex mtx_;  // mutex for thread-safe access

  rmm::cuda_device_id device_id_{rmm::get_current_cuda_device()};
};  // namespace detail

}  // namespace mr::detail
RMM_NAMESPACE_END
