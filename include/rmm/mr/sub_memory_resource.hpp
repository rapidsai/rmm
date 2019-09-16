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

#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>

namespace rmm {
namespace mr {

namespace detail {
  
  template <typename T>
  inline T round_up_safe(T number_to_round, T modulus) {
    T remainder = number_to_round % modulus;
    if (remainder == 0) { return number_to_round; }
    T rounded_up = number_to_round - remainder + modulus;
    if (rounded_up < number_to_round) {
      throw std::invalid_argument("Attempt to round up beyond the type's maximum value");
    }
    return rounded_up;
  }

} // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Memory resource that allocates/deallocates using the cnmem pool sub-allocator
 * the cnmem pool sub-allocator for allocation/deallocation.
 *---------------------------------------------------------------------------**/
class sub_memory_resource final : public device_memory_resource {
 public:

  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  static constexpr size_t allocation_alignment = 256;

  /**---------------------------------------------------------------------------*
   * @brief Construct a cnmem memory resource and allocate the initial device
   * memory pool

   * TODO Add constructor arguments for other CNMEM options/flags
   *
   * @param initial_pool_size Size, in bytes, of the intial pool size. When
   * zero, an implementation defined pool size is used.
   *---------------------------------------------------------------------------**/
  explicit sub_memory_resource(std::size_t initial_pool_size = default_initial_size, 
                               std::size_t maximum_pool_size = default_maximum_size)
    : heap_resource{rmm::mr::get_default_resource()}
  {
    if (initial_pool_size == default_initial_size)  {
      int device{0};
      cudaGetDevice(&device);
      int memsize{0};
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, device);
      initial_pool_size = props.totalGlobalMem / 2;

      if (maximum_pool_size == default_maximum_size)
        maximum_pool_size = props.totalGlobalMem;
    }

    // Allocate initial block
    no_sync_blocks.insert(block_from_heap(initial_pool_size, 0));

    // TODO device handling?

    // TODO allocation should check maximum pool size

    // TODO smarter new block size heuristic
  }

  ~sub_memory_resource() {
    free_all();
  }

  bool supports_streams() const noexcept override { return true; }

 private:

  struct block
  {
    char* ptr;
    size_t size;

    bool operator<(const block& rhs) const { 
      if (size < rhs.size) return ptr < rhs.ptr;
      return false;
    }
  };

  using block_set = std::set<block>;

  inline block next_larger_block(block_set &blocks, size_t size)
  {
    block dummy{0, size};
    auto iter = blocks.lower_bound(dummy);

    if (iter != blocks.end())
    {
      block found = *iter;
      blocks.erase(iter);
      return found;
    }
    
    return dummy;
  }

  inline block available_larger_block(size_t size, cudaStream_t stream)
  {
    // Try to find a larger block that doesn't require syncing
    block b = next_larger_block(no_sync_blocks, size);

    // Try to find a larger block in a different stream
    if (b.ptr == nullptr) {
      for (auto s : sync_blocks) {
        if (s.first != stream) b = next_larger_block(s.second, size);
        if (b.ptr != nullptr) {
          // synchronize the stream associated with this list
          cudaError_t result = cudaStreamSynchronize(s.first);
          
          if (result != cudaErrorInvalidResourceHandle && // stream deleted
              result != cudaSuccess)                      // stream synced
            throw std::runtime_error{"cudaStreamSynchronize failure"};

          // insert all other blocks into the no_sync list
          no_sync_blocks.insert(std::make_move_iterator(s.second.begin()),
                                std::make_move_iterator(s.second.end()));

          // remove this stream from the freelist
          sync_blocks.erase(stream);
          return b;
        }
      }

      // no larger blocks waiting on other streams, so create one
      b = block_from_heap(size, stream);
    }

    
    return b;
  }

  inline block split_block(block b, size_t size)
  {
    if (b.size > size)
    {
      block rest{b.ptr + size, b.size - size};
      no_sync_blocks.insert(rest);
      b.size = size;
    }

    return b;
  }

  inline block merge_blocks(const block& a, const block& b)
  {
    if (a.ptr + a.size != b.ptr)
      throw std::logic_error("Invalid block merge");
    
    return block{a.ptr, a.size + b.size};
  }

  inline void* allocate_from_block(const block& b, size_t size)
  {
    if (b.ptr == nullptr) throw std::bad_alloc{};
    block split = split_block(b, size);
    allocated_blocks[split.ptr] = split;
    return reinterpret_cast<void*>(split.ptr);
  }

  // check if next / prev are in blocks and merge if so
  inline void free_block(const block& b, block_set& blocks) {
    block_set::iterator prev{blocks.end()}, next{blocks.end()};

    // TODO this is a linear search since the set is ordered by size
    // would be nice to have a faster way
    for (auto iter = blocks.begin(); iter != blocks.end(); ++iter) {
      if (iter->ptr + iter->size == b.ptr) prev = iter;
      else if (b.ptr + b.size == iter->ptr) { next = iter; break; }
    }

    block merged = b;
    if (prev != blocks.end()) {
      merged = merge_blocks(*prev, merged);  
      blocks.erase(prev);
    }

    if (next != blocks.end()) {
      merged = merge_blocks(merged, *next);
      blocks.erase(next);
    }

    blocks.insert(merged);
  }

  inline void find_and_free_block(void *p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    auto i = allocated_blocks.find(reinterpret_cast<char*>(p));
    block b{};
    if (i != allocated_blocks.end()) { // found
      b = i->second;
      assert(b.size == detail::round_up_safe(size, allocation_alignment));
      allocated_blocks.erase(i);
      free_block(b, sync_blocks[stream]);
    }
    else {
      throw std::runtime_error("Pointer not allocated by this resource.");
    }
  }

  inline block block_from_heap(size_t size, cudaStream_t stream)
  {
    void* p = heap_resource->allocate(size, stream);
    if (p == nullptr) throw std::bad_alloc{};
    block b{reinterpret_cast<char*>(p), size};
    heap_blocks[b.ptr] = b;
    return b;
  }

  inline void free_all()
  {
    for (auto b : heap_blocks) heap_resource->deallocate(b.first, b.second.size);
    heap_blocks.clear();
  }

  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cnmem.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    bytes = detail::round_up_safe(bytes, allocation_alignment);
    block b = available_larger_block(bytes, stream);
    return allocate_from_block(b, bytes);
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
  }

  /**---------------------------------------------------------------------------*
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   *---------------------------------------------------------------------------**/
  std::pair<size_t,size_t> do_get_mem_info( cudaStream_t stream){
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

  size_t maximum_pool_size{default_maximum_size};
  std::set<cudaStream_t> registered_streams{};
  std::mutex streams_mutex{};

  device_memory_resource *heap_resource;

  // no_sync_free_list: list of unencumbered blocks
  // no need to sync a stream to allocate from this list
  // TODO: what container? Should be sorted by size? Locality? Binned?
  // A free_list is likely to become a class
  block_set no_sync_blocks;

  // sync_free_lists: map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list
  std::map<cudaStream_t, block_set> sync_blocks;

  // allocated_blocks: map of allocated [ptr, block] pairs
  std::map<char*, block> allocated_blocks;

  // blocks allocated from heap: so they can be easily freed
  std::map<char*, block> heap_blocks;
};

}  // namespace mr
}  // namespace rmm
