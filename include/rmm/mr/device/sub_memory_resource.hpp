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

#include <cuda_runtime_api.h>
#include <exception>
#include <iostream>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <mutex>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief Memory resource that allocates/deallocates using a pool sub-allocator
 *---------------------------------------------------------------------------**/
class sub_memory_resource final : public device_memory_resource {
 public:

  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  static constexpr size_t allocation_alignment = 256;

  /**---------------------------------------------------------------------------*
   * @brief Construct a suballocator memory resource and allocate the initial
   * device memory pool
   *
   * @param initial_pool_size Size, in bytes, of the initial pool size. When
   * zero, an implementation defined pool size is used.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to.
   *---------------------------------------------------------------------------**/
  explicit sub_memory_resource(std::size_t initial_pool_size = default_initial_size,
                               std::size_t maximum_pool_size = default_maximum_size)
    : heap_resource{rmm::mr::get_default_resource()}
  {
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
    // TODO: non-default stream?
    no_sync_blocks.insert(block_from_heap(initial_pool_size, 0));

    // TODO thread safety

    // TODO device handling?

    // TODO allocation should check maximum pool size

    // TODO smarter new block size heuristic
  }

  ~sub_memory_resource() {
    free_all();
  }

  bool supports_streams() const noexcept override { return true; }

 private:

  using block = rmm::mr::detail::block;
  using free_list = rmm::mr::detail::free_list<>;

  block block_from_sync_list(size_t size, cudaStream_t stream)
  {
    free_list& blocks = sync_blocks.at(stream);
    block b = blocks.best_fit(size);
    if (b.ptr != nullptr) { // found one
      cudaError_t result = cudaStreamSynchronize(stream);

      if (result != cudaErrorInvalidResourceHandle && // stream deleted
          result != cudaSuccess)                      // stream synced
        throw std::runtime_error{"cudaStreamSynchronize failure"};

      // insert all other blocks into the no_sync list
      no_sync_blocks.insert(blocks.begin(), blocks.end());
      blocks.clear();

      // remove this stream from the freelist
      sync_blocks.erase(stream);
    }
    return b;
  }

  block available_larger_block(size_t size, cudaStream_t stream) {
    // Try to find a larger block that doesn't require syncing
    block b = no_sync_blocks.best_fit(size);

    // nothing in no-sync free list, look for one on a stream
    if (b.ptr == nullptr) {

      // Try to find a larger block in a different stream
      for (auto s : sync_blocks) {
        if (s.first != stream) b = block_from_sync_list(size, s.first);
      }

      // nothing available in other streams, look in current stream's list
      if (b.ptr == nullptr && (sync_blocks.find(stream) != sync_blocks.end())) {
          b = block_from_sync_list(size, stream);
      }

      // no larger blocks waiting on other streams, so create one
      if (b.ptr == nullptr) b = block_from_heap(size, stream);
    }

    return b;
  }

  void* allocate_from_block(block const& b, size_t size)
  {
    if (b.ptr == nullptr)
      throw std::bad_alloc{};

    block alloc{b};

    if (b.size > size)
    {
      block rest{b.ptr + size, b.size - size, false};
      no_sync_blocks.insert(rest);
      alloc.size = size;
    }

    allocated_blocks.insert(alloc);
    return reinterpret_cast<void*>(alloc.ptr);
  }


  void find_and_free_block(void *p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    auto i = allocated_blocks.find(block{static_cast<char*>(p)});

    if (i != allocated_blocks.end()) { // found
      //assert(i->size == rmm::detail::align_up(size, allocation_alignment));
      sync_blocks[stream].insert(*i);
      allocated_blocks.erase(i);
    }
    else {
      throw std::runtime_error("Pointer not allocated by this resource.");
    }
  }

  block block_from_heap(size_t size, cudaStream_t stream)
  {
    void* p = heap_resource->allocate(size, stream);
    if (p == nullptr)
      throw std::bad_alloc{};
    block b{reinterpret_cast<char*>(p), size, true};
    heap_blocks[b.ptr] = b;
    return b;
  }

  void free_all()
  {
    for (auto b : heap_blocks) heap_resource->deallocate(b.first, b.second.size);
    heap_blocks.clear();
  }

#ifndef NDEBUG
  void print() {
    std::cout << "allocated_blocks: " << allocated_blocks.size() << "\n";
    for (auto b : allocated_blocks) { b.print(); }

    std::cout << "no-sync free blocks: ";
    no_sync_blocks.print();

    std::cout << "sync free blocks: ";
    for (auto s : sync_blocks) { 
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
    void* p = allocate_from_block(b, bytes);
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

  size_t maximum_pool_size{default_maximum_size};

  device_memory_resource *heap_resource;

  // no-sync free list: list of unencumbered blocks
  // no need to sync a stream to allocate from this list
  free_list no_sync_blocks;

  // sync free lists: map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list
  std::unordered_map<cudaStream_t, free_list> sync_blocks;

  //std::list<block> allocated_blocks;
  std::set<block> allocated_blocks;


  // blocks allocated from heap: so they can be easily freed
  std::unordered_map<char*, block> heap_blocks;
};

}  // namespace mr
}  // namespace rmm
