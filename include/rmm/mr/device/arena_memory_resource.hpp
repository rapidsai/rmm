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
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <map>
#include <shared_mutex>

namespace rmm {
namespace mr {

/**
 * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * GPU memory is divided into a global arena, per-thread arenas for default streams, and per-stream
 * arenas for non-default streams. Each arena allocates memory from the global arena in chunks
 * called superblocks.
 *
 * Blocks in each arena are allocated using address-ordered first fit. When a block is freed, it is
 * coalesced with neighbouring free blocks if the addresses are contiguous. Free superblocks are
 * returned to the global arena.
 *
 * In real-world applications, allocation sizes tend to follow a power law distribution in which
 * large allocations are rare, but small ones quite common. By handling small allocations in the
 * per-thread arena, adequate performance can be achieved without introducing excessive memory
 * fragmentation under high concurrency.
 *
 * This design is inspired by several existing CPU memory allocators targeting multi-threaded
 * applications (glibc malloc, Hoard, jemalloc, TCMalloc), albeit in a simpler form. Possible future
 * improvements include using size classes, allocation caches, and more fine-grained locking or
 * lock-free approaches.
 *
 * \see Wilson, P. R., Johnstone, M. S., Neely, M., & Boles, D. (1995, September). Dynamic storage
 * allocation: A survey and critical review. In International Workshop on Memory Management (pp.
 * 1-116). Springer, Berlin, Heidelberg.
 * \see Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. (2000). Hoard: A scalable
 * memory allocator for multithreaded applications. ACM Sigplan Notices, 35(11), 117-128.
 * \see Evans, J. (2006, April). A scalable concurrent malloc (3) implementation for FreeBSD. In
 * Proc. of the bsdcan conference, ottawa, canada.
 * \see https://sourceware.org/glibc/wiki/MallocInternals
 * \see http://hoard.org/
 * \see http://jemalloc.net/
 * \see https://github.com/google/tcmalloc
 *
 * @tparam Upstream Memory resource to use for allocating memory for the global arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Construct an `arena_memory_resource`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`.
   * @throws rmm::logic_error if `initial_size` is neither the default nor aligned to a multiple of
   * 256 bytes.
   * @throws rmm::logic_error if `maximum_size` is neither the default nor aligned to a multiple of
   * 256 bytes.
   *
   * @param upstream_mr The memory resource from which to allocate blocks for the pool
   * @param initial_size Minimum size, in bytes, of the initial global arena. Defaults to half of
   * the available memory on the current device.
   * @param maximum_size Maximum size, in bytes, that the global arena can grow to. Defaults to all
   * of the available memory on the current device.
   */
  explicit arena_memory_resource(Upstream* upstream_mr,
                                 std::size_t initial_size = global_arena::default_initial_size,
                                 std::size_t maximum_size = global_arena::default_maximum_size)
    : global_arena_{upstream_mr, initial_size, maximum_size}
  {
  }

  // Disable copy (and move) semantics.
  arena_memory_resource(arena_memory_resource const&) = delete;
  arena_memory_resource& operator=(arena_memory_resource const&) = delete;

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
   * @return bool false.
   */
  bool supports_get_mem_info() const noexcept override { return false; }

 private:
  using global_arena = detail::arena::global_arena<Upstream>;
  using arena        = detail::arena::arena<Upstream>;
  using read_lock    = std::shared_lock<std::shared_timed_mutex>;
  using write_lock   = std::lock_guard<std::shared_timed_mutex>;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256-byte alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled.
   *
   * @param bytes The size in bytes of the allocation.
   * @param stream The stream to associate this allocation with.
   * @return void* Pointer to the newly allocated memory.
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    if (bytes <= 0) return nullptr;

    bytes = detail::arena::align_up(bytes);
    return get_arena(stream).allocate(bytes);
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   */
  void do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) override
  {
    if (p == nullptr || bytes <= 0) return;

    bytes = detail::arena::align_up(bytes);
    if (!get_arena(stream).deallocate(p, bytes, stream)) {
      deallocate_from_other_arena(p, bytes, stream);
    }
  }

  /**
   * @brief Deallocate memory pointed to by `p` that was allocated in a different arena.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   */
  void deallocate_from_other_arena(void* p, std::size_t bytes, cuda_stream_view stream)
  {
    stream.synchronize_no_throw();

    read_lock lock(mtx_);

    if (use_per_thread_arena(stream)) {
      auto const id = std::this_thread::get_id();
      for (auto& kv : thread_arenas_) {
        // If the arena does not belong to the current thread, try to deallocate from it, and return
        // if successful.
        if (kv.first != id && kv.second->deallocate(p, bytes)) return;
      }
    } else {
      for (auto& kv : stream_arenas_) {
        // If the arena does not belong to the current stream, try to deallocate from it, and return
        // if successful.
        if (stream != kv.first && kv.second.deallocate(p, bytes)) return;
      }
    }

    // The thread that originally allocated the block has terminated, deallocate directly in the
    // global arena.
    global_arena_.deallocate({p, bytes});
  }

  /**
   * @brief Get the arena associated with the current thread or the given stream.
   *
   * @param stream The stream associated with the arena.
   * @return arena& The arena associated with the current thread or the given stream.
   */
  arena& get_arena(cuda_stream_view stream)
  {
    if (use_per_thread_arena(stream)) {
      return get_thread_arena();
    } else {
      return get_stream_arena(stream);
    }
  }

  /**
   * @brief Get the arena associated with the current thread.
   *
   * @return arena& The arena associated with the current thread.
   */
  arena& get_thread_arena()
  {
    auto const id = std::this_thread::get_id();
    {
      read_lock lock(mtx_);
      auto const it = thread_arenas_.find(id);
      if (it != thread_arenas_.end()) { return *it->second; }
    }
    {
      write_lock lock(mtx_);
      auto a = std::make_shared<arena>(global_arena_);
      thread_arenas_.emplace(id, a);
      thread_local detail::arena::arena_cleaner<Upstream> cleaner{a};
      return *a;
    }
  }

  /**
   * @brief Get the arena associated with the given stream.
   *
   * @return arena& The arena associated with the given stream.
   */
  arena& get_stream_arena(cuda_stream_view stream)
  {
    RMM_LOGGING_ASSERT(!use_per_thread_arena(stream));
    {
      read_lock lock(mtx_);
      auto const it = stream_arenas_.find(stream.value());
      if (it != stream_arenas_.end()) { return it->second; }
    }
    {
      write_lock lock(mtx_);
      stream_arenas_.emplace(stream.value(), global_arena_);
      return stream_arenas_.at(stream.value());
    }
  }

  /**
   * @brief Get free and available memory for memory resource.
   *
   * @param stream to execute on.
   * @return std::pair containing free_size and total_size of memory.
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    return std::make_pair(0, 0);
  }

  /**
   * @brief Should a per-thread arena be used given the CUDA stream.
   *
   * @param stream to check.
   * @return true if per-thread arena should be used, false otherwise.
   */
  static bool use_per_thread_arena(cuda_stream_view stream)
  {
    return stream.is_per_thread_default();
  }

  /// The global arena to allocate superblocks from.
  global_arena global_arena_;
  /// Arenas for default streams, one per thread.
  /// Implementation note: for small sizes, map is more efficient than unordered_map.
  std::map<std::thread::id, std::shared_ptr<arena>> thread_arenas_;
  /// Arenas for non-default streams, one per stream.
  /// Implementation note: for small sizes, map is more efficient than unordered_map.
  std::map<cudaStream_t, arena> stream_arenas_;
  /// Mutex for read and write locks.
  mutable std::shared_timed_mutex mtx_;
};

}  // namespace mr
}  // namespace rmm
