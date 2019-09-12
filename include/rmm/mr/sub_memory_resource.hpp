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
    // TODO allocate initial block

    // TODO device handling?

    // TODO allocation should check maximum pool size

    // TODO smarter new block size heuristic
  }

  ~sub_memory_resource() {
  }

  bool supports_streams() const noexcept override { return true; }

 private:

  struct block
  {
    intptr_t ptr;
    size_t size;

    bool operator==(block const* rhs) const {
      return this->ptr == rhs->ptr && this->size == rhs->size;
    }
    bool operator<(const block *rhs) const { return this->size < rhs->size; }
  };

  using block_set = std::set<block*>;

  inline block* next_larger_block(block_set &blocks, size_t size)
  {
    block dummy{0, size};
    auto iter = blocks.lower_bound(&dummy);
    
    return iter != blocks.end() ? *iter : nullptr;
  }

  inline block* available_larger_block(size_t size, cudaStream_t stream)
  {
    // Try to find a larger block that doesn't require syncing
    block* b = next_larger_block(no_sync_blocks, size);

    // Try to find a larger block in a different stream
    if (!b) {
      for (auto s : sync_blocks) {
        if (s.first != stream) b = next_larger_block(s.second, size);
        if (b) {
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
          break;
        }
      }
    }

    // no larger blocks available, create one
    b = block_from_heap(size, stream);

    return b;
  }

  inline block* split_block(block *b, size_t size, cudaStream_t stream)
  {
    if (b->size > size)
    {
      block *rest = new block{static_cast<intptr_t>(b->ptr+size), b->size - size};
      b->size = size;
      no_sync_blocks.insert(rest);
    }

    return b;
  }

  inline intptr_t allocate_from_block(block *b, size_t size, cudaStream_t stream)
  {
    if (b == nullptr) throw std::bad_alloc{};
    b = split_block(b, size, stream);
    allocated_blocks[b->ptr] = b;
    return b->ptr;
  }

  inline void find_and_deallocate_block(void *p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    auto i = allocated_blocks.find(reinterpret_cast<intptr_t>(p));
    block *b = nullptr;
    if (i != allocated_blocks.end()) { // found
      b = std::move(i->second);
      assert(b->size == detail::round_up_safe(size, allocation_alignment));
      allocated_blocks.erase(i);
      sync_blocks[stream].insert(b);
    }
    else {
      throw std::runtime_error("Pointer not allocated by this resource.");
    }
  }

  inline block* block_from_heap(size_t size, cudaStream_t stream)
  {
    void* p = heap_resource->allocate(size, stream);
    if (p != nullptr)
      return new block{reinterpret_cast<intptr_t>(p), size};
    else return nullptr;
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
    block *b = available_larger_block(bytes, stream);
    return reinterpret_cast<void*>(allocate_from_block(b, bytes, stream));
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws std::runtime_error if \p p was not allocated by this resource.
   *
   * @param p Pointer to be deallocated
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    find_and_deallocate_block(p, bytes, stream);
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
  std::map<std::intptr_t, block*> allocated_blocks;
};

}  // namespace mr
}  // namespace rmm
