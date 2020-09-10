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

#include <mutex>
#include <unordered_map>

namespace rmm {
namespace mr {

/**
 * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * GPU memory is divided into a global heap and per-thread heaps. Each thread allocates memory from
 * the global heap in chunks called superblocks. All superblocks are the same size. Objects larger
 * than half the size of a superblock are managed directly using the global heap.
 *
 * Blocks in the per-thread heap are allocated using address-ordered first-fit. When a block is
 * freed, it is coalesced with neighbouring free blocks if the addresses are contiguous and do not
 * cross superblock boundaries. Completely empty superblocks are returned to the global heap.
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
 * @tparam UpstreamResource memory_resource to use for allocating the arenas. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena_memory_resource final : public device_memory_resource {
 public:
  // The required alignment of this allocator.
  static constexpr size_t allocation_alignment = 256;

  /**
   * @brief Construct an `arena_memory_resource`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate memory for the arenas.
   */
  explicit arena_memory_resource(Upstream* upstream_mr) : upstream_mr_{upstream_mr}
  {
    RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");
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
   * @return bool false
   */
  bool supports_get_mem_info() const noexcept override { return false; }

 private:
  using arena      = detail::arena::arena<Upstream>;
  using lock_guard = std::lock_guard<std::mutex>;

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

      if (arena::handles_size(bytes)) { return get_arena()->allocate(bytes); }
    }
#endif
    return upstream_mr_->allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws rmm::bad_alloc if `p` is not found
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    if (p == nullptr || bytes <= 0) return;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (stream == cudaStreamDefault || stream == cudaStreamPerThread) {
      bytes = rmm::detail::align_up(bytes, allocation_alignment);
      if (arena::handles_size(bytes)) {
        if (!get_arena()->deallocate(p, bytes, stream)) {
          deallocate_across_arenas(p, bytes, stream);
        }
        return;
      }
    }
#endif
    upstream_mr_->deallocate(p, bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by `p` that was allocated in a different arena.
   *
   * @throws rmm::bad_alloc if `p` is not found
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate_across_arenas(void* p, size_t bytes, cudaStream_t stream)
  {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    auto id = std::this_thread::get_id();
    {
      lock_guard lock(mtx_);
      for (auto& kv : arenas_) {
        if (kv.first != id && kv.second->deallocate(p, bytes)) return;
      }
    }

    RMM_FAIL("Allocation not found", rmm::bad_alloc);
  }

  /**
   * @brief Get free and available memory for memory resource.
   *
   * @param stream to execute on
   * @return std::pair containing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    return std::make_pair(0, 0);
  }

  /**
   * @brief Get the arena associated with the current thread.
   *
   * @return std::shared_ptr<arena> The arena associated with the current thread
   */
  std::shared_ptr<arena> get_arena()
  {
    thread_local auto a = std::make_shared<arena>(upstream_mr_);
    thread_local bool is_initialized{false};

    if (!is_initialized) {
      lock_guard lock(mtx_);
      arenas_.emplace(std::this_thread::get_id(), a);
      is_initialized = true;
    }
    return a;
  }

  Upstream* upstream_mr_;
  std::unordered_map<std::thread::id, std::shared_ptr<arena>> arenas_;
  mutable std::mutex mtx_;
};

}  // namespace mr
}  // namespace rmm
