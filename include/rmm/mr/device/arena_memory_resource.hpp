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

#include <shared_mutex>
#include <unordered_map>

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
  static constexpr size_t allocation_alignment = 256;

  /**
   * @brief Construct an `arena_memory_resource`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate memory for the arenas.
   */
  explicit arena_memory_resource(Upstream* upstream_mr) : upstream_mr_{upstream_mr} {}

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
  using read_lock  = std::shared_lock<std::shared_timed_mutex>;
  using write_lock = std::lock_guard<std::shared_timed_mutex>;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource.
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `size_t`.
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  void deallocate_across_arena(void* p, size_t bytes, cudaStream_t stream)
  {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    write_lock lock(mtx_);
    for (auto& kv : arenas_) {
      if (kv.second.deallocate(p, bytes)) return;
    }

    RMM_FAIL("Allocation not found", rmm::bad_alloc);
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

      if (bytes < arena::maximum_allocation_size) { return get_arena().allocate(bytes); }
    }
#endif
    return upstream_mr_->allocate(bytes, stream);
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
      if (bytes < arena::maximum_allocation_size) {
        if (!get_arena().deallocate(p, bytes, stream)) {
          deallocate_across_arena(p, bytes, stream);
        }
        return;
      }
    }
#endif
    return upstream_mr_->deallocate(p, bytes, stream);
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

  arena& get_arena()
  {
    auto id = std::this_thread::get_id();
    {
      read_lock lock(mtx_);
      auto it = arenas_.find(id);
      if (it != arenas_.end()) { return it->second; }
    }
    {
      write_lock lock(mtx_);
      arenas_.emplace(id, upstream_mr_);
      return arenas_.at(id);
    }
  }

  Upstream* upstream_mr_;
  std::unordered_map<std::thread::id, arena> arenas_;
  mutable std::shared_timed_mutex mtx_;
};

}  // namespace mr
}  // namespace rmm
