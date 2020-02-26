/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/detail/free_list.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <algorithm>
#include <mutex>

namespace rmm {
namespace mr {

/**
 * @brief Memory resource that allocates/deallocates using a pool sub-allocator
 */
template <typename UpstreamResource>
class pool_memory_resource final : public device_memory_resource {
 public:

  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  static constexpr size_t allocation_alignment = 256;

  /**
   * @brief Construct a suballocator memory resource and allocate the initial
   * device memory pool
   *
   * @param initial_pool_size Size, in bytes, of the initial pool size. When
   * zero, an implementation defined pool size is used.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to.
   */

  explicit pool_memory_resource(
      UpstreamResource* upstream_mr,
      std::size_t initial_pool_size = default_initial_size,
      std::size_t maximum_pool_size = default_maximum_size)
      : upstream_mr_{upstream_mr} {
    cudaDeviceProp props;

    if (initial_pool_size == default_initial_size)  {
      int device{0};
      cudaGetDevice(&device);
      int memsize{0};
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, device);
      initial_pool_size = props.totalGlobalMem / 2;
    }

    initial_pool_size = rmm::detail::align_up(initial_pool_size, allocation_alignment);

    if (maximum_pool_size == default_maximum_size)
        maximum_pool_size = props.totalGlobalMem;

    // Allocate initial block
    stream_blocks_[0].insert(block_from_upstream(initial_pool_size, 0));

    // TODO device handling?

    // TODO allocation should check maximum pool size

    // TODO smarter new block size heuristic
  }

  ~pool_memory_resource() {
    free_all();
  }

  bool supports_streams() const noexcept override { return true; }

  UpstreamResource* get_upstream() const noexcept { return upstream_mr_; }

 private:

  using block = rmm::mr::detail::block;
  using free_list = rmm::mr::detail::free_list<>;

  block block_from_sync_list(free_list &blocks, size_t size, 
                             cudaStream_t list_stream, cudaStream_t stream)
  {
    block b = blocks.best_fit(size);
    if (b.ptr != nullptr) { // found one
      if (list_stream != stream) {
        cudaError_t result = cudaStreamSynchronize(list_stream);

        RMM_EXPECTS((result == cudaSuccess ||                   // stream synced
                     result == cudaErrorInvalidResourceHandle), // stream deleted
                    rmm::cuda_error, "cudaStreamSynchronize failure");

        // insert all other blocks into this stream's list
        // TODO: Should we do this? This could cause thrashing between two 
        // streams. On the other hand, this can also reduce fragmentation.
        stream_blocks_[stream].insert(blocks.begin(), blocks.end());
        blocks.clear();

        // remove this stream from the freelist
        stream_blocks_.erase(list_stream);
      }
    }
    return b;
  }

  block available_larger_block(size_t size, cudaStream_t stream) {
    block b{};

    // Try to find a larger block in the same stream
    // Try to find a block in the same stream
    auto iter = stream_blocks_.find(stream);
    if (iter != stream_blocks_.end())
      b = block_from_sync_list(iter->second, size, stream, stream);

    // nothing in this stream's free list, look for one on another stream
    if (b.ptr == nullptr) {
      auto s = stream_blocks_.begin();
      while (b.ptr == nullptr && s != stream_blocks_.end()) {
        if (s->first != stream) 
          b = block_from_sync_list(s->second, size, s->first, stream);
        s++;
      }
    }

    // no larger blocks waiting on other streams, so create one
    if (b.ptr == nullptr) b = block_from_upstream(size, stream);

    return b;
  }

  void* allocate_from_block(block const& b, size_t size, cudaStream_t stream)
  {
    block alloc{b};

    if (b.size > size)
    {
      block rest{b.ptr + size, b.size - size, false};
      stream_blocks_[stream].insert(rest);
      alloc.size = size;
    }

    allocated_blocks_.insert(alloc);
    return reinterpret_cast<void*>(alloc.ptr);
  }


  void find_and_free_block(void *p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    auto i = allocated_blocks_.find(block{static_cast<char*>(p)});

    if (i != allocated_blocks_.end()) {  // found
      // assert(i->size == rmm::detail::align_up(size, allocation_alignment));
      stream_blocks_[stream].insert(*i);
      allocated_blocks_.erase(i);
    } else {
      throw std::runtime_error("Pointer not allocated by this resource.");
    }
  }

  block block_from_upstream(size_t size, cudaStream_t stream)
  {
    void* p = upstream_mr_->allocate(size, stream);
    block b{reinterpret_cast<char*>(p), size, true};
    upstream_blocks_[b.ptr] = b;
    return b;
  }

  void free_all()
  {
    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.first, b.second.size);
    upstream_blocks_.clear();
  }

#ifndef NDEBUG
  void print() {
    std::size_t free, total;
    std::tie(free, total) = upstream_mr_->get_mem_info(0);
    std::cout << "GPU free memory: " << free << "total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto h : upstream_blocks_) { 
      h.second.print();
      upstream_total += h.second.size;
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
    for (auto b : allocated_blocks_) { b.print(); }

    std::cout << "sync free blocks: ";
    for (auto s : stream_blocks_) { 
      std::cout << "stream " << s.first << " ";
      s.second.print();
    }
    std::cout << "\n";
  }
#endif // DEBUG

  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @param The stream to associate this allocation with
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    bytes = rmm::detail::align_up(bytes, allocation_alignment);
    block b = available_larger_block(bytes, stream);
    void* p = allocate_from_block(b, bytes, stream);
#ifndef NDEBUG
    //std::cout << "Allocate " << bytes << " B on stream " << stream << "\n";
    //print();
#endif
    return p;
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws std::runtime_error if \p p was not allocated by this resource.
   *
   * @param p Pointer to be deallocated
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    find_and_free_block(p, bytes, stream);
#ifndef NDEBUG
    //std::cout << "Free " << bytes << " B on stream " << stream 
    //          << " pointer: " << p << "\n";
    //print();
#endif
  }

  /**---------------------------------------------------------------------------*
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   *---------------------------------------------------------------------------**/
  std::pair<size_t,size_t> do_get_mem_info( cudaStream_t stream) const override {
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

  size_t maximum_pool_size_{default_maximum_size};

  UpstreamResource* upstream_mr_;

  // map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list to a different stream
  std::map<cudaStream_t, free_list> stream_blocks_;

  //std::list<block> allocated_blocks;
  std::set<block> allocated_blocks_;

  // blocks allocated from upstream heap: so they can be easily freed
  std::map<char*, block> upstream_blocks_;
};

}  // namespace mr
}  // namespace rmm
