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
//#include <cassert>
#include <exception>
#include <iostream>
#include <mutex>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <chrono>
#include <map>

namespace {

// double precision time durations
template <typename T>
std::chrono::duration<double, std::micro>  microseconds(T&& t)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t);
}

std::map<int, double> counters;
std::string TIMER_IDS[] = {"available_larger_block", 
                           "block_from_sync_list",
                           "find_and_free_block",
                           "get_best_fit",
                           "insert_and_merge"};

struct raii_timer {
  using timer = std::chrono::high_resolution_clock;
  raii_timer(int id) : id(id), start(timer::now()) {}
  ~raii_timer() {
    auto end = timer::now();
    auto diff = end - start;

    counters[id] += (double)microseconds(diff).count();
  }

  int id;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

void print_timers() {
  std::cout << "----  Timers  ----\n";
  for (auto x : counters) {
    std::cout << TIMER_IDS[x.first] << ": " << x.second << " us\n";
  }
  std::cout << "----  ------  ----\n\n";
}

}


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

    initial_pool_size = detail::round_up_safe(initial_pool_size,
                                              allocation_alignment);

    if (maximum_pool_size == default_maximum_size)
        maximum_pool_size = props.totalGlobalMem;

    // Allocate initial block
    // TODO: non-default stream?
    no_sync_blocks.insert_and_merge(block_from_heap(initial_pool_size, 0));

    // TODO thread safety

    // TODO device handling?

    // TODO allocation should check maximum pool size

    // TODO smarter new block size heuristic
  }

  ~sub_memory_resource() {
    print_timers();
    counters.clear();
    free_all();
#ifndef NDEBUG
    /*std::cout << "Statistics\n"
              << "#inserts: " << num_inserts << "\n"
              << "#erases: " << num_erases << "\n";*/
#endif
  }

  bool supports_streams() const noexcept override { return true; }

 private:

  struct block
  {
    char* ptr;          ///< Raw memory pointer
    size_t size;        ///< Size in bytes
    bool is_head_block; ///< Indicates whether ptr was allocated from the heap

    bool operator<(block const& rhs) const { return ptr < rhs.ptr; };

    void print() const {
      std::cout << reinterpret_cast<void*>(ptr) << " " << size << "B\n";
    }
  };

  struct free_list {

    inline block get_best_fit(size_t size) {
      raii_timer(3);
      block dummy{nullptr, size, false};
      // find best fit block
      auto iter = std::min_element(blocks.begin(), blocks.end(),
        [size](block lhs, block rhs) {
          return (lhs.size >= size) && 
                 ((lhs.size < rhs.size) || (rhs.size < size));
        });

      if (iter->size >= size)
      {
        block found = *iter;
        remove_block(iter);
        return found;
      }
      
      return dummy;
    }

    inline void insert_and_merge(block const& b) {
      raii_timer(4);
      auto next = std::find_if(blocks.begin(), blocks.end(),
        [b](block const& i) { return i.ptr > b.ptr; });

      auto previous = std::prev(next);
      bool merge_prev = !b.is_head_block && (previous->ptr + previous->size == b.ptr); 
      bool merge_next = !next->is_head_block && (b.ptr + b.size == next->ptr);

      if (merge_prev) {
        *previous = merge_blocks(*previous, b);
        if (merge_next) {
          *previous = merge_blocks(*previous, *next);
          blocks.erase(next);
        }
      } else if (merge_next) {
        *next = merge_blocks(b, *next);
      } else {
        blocks.insert(next, b);
      }
    }

    inline void merge(free_list&& other) {
      for (auto iter = other.blocks.begin(); iter != other.blocks.end(); ++iter) {
        insert_and_merge(*iter);
      }
      other.blocks.clear();
    }

    void print() const {
      std::cout << blocks.size() << "\n";
      for (block const& b : blocks) { b.print(); }
    }

  private:
    using block_list = std::list<block>;

    /* void insert_block(block const& b) {
      blocks.push_back(b);
#ifndef NDEBUG
      num_inserts++;
      std::cout << "Inserted " << b.size << " set size: "
                << blocks.size() << "\n";
#endif
    }*/

    inline void remove_block(block_list::iterator const& b) {
      blocks.erase(b);
#ifndef NDEBUG
      num_erases++;
#endif
    }

    block_list blocks;
    //std::mutex blocks_mutex;

    // stats
    size_t num_inserts{0};
    size_t num_erases{0};
  };

  inline block block_from_sync_list(size_t size, cudaStream_t stream)
  {
    raii_timer(1);

    free_list& blocks = sync_blocks.at(stream);
    block b = blocks.get_best_fit(size);
    if (b.ptr != nullptr) { // found one
      cudaError_t result = cudaStreamSynchronize(stream);

      if (result != cudaErrorInvalidResourceHandle && // stream deleted
          result != cudaSuccess)                      // stream synced
        throw std::runtime_error{"cudaStreamSynchronize failure"};

      // insert all other blocks into the no_sync list
      no_sync_blocks.merge(std::move(blocks));

      // remove this stream from the freelist
      sync_blocks.erase(stream);
    }
    return b;
  }

  inline block available_larger_block(size_t size, cudaStream_t stream)
  {
    raii_timer(0);

    // Try to find a larger block that doesn't require syncing
    block b = no_sync_blocks.get_best_fit(size);

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

  inline block split_block(block b, size_t size)
  {
    if (b.size > size)
    {
      block rest{b.ptr + size, b.size - size};
      no_sync_blocks.insert_and_merge(rest);
      b.size = size;
    }

    return b;
  }

  static inline block merge_blocks(block const& a, block const& b)
  {
    if (a.ptr + a.size != b.ptr || b.is_head_block)
      throw std::logic_error("Invalid block merge");
    
    return block{a.ptr, a.size + b.size};
  }

  inline void* allocate_from_block(block const& b, size_t size)
  {
    if (b.ptr == nullptr)
      throw std::bad_alloc{};
    block split = split_block(b, size);
    allocated_blocks.push_back(split);
    return reinterpret_cast<void*>(split.ptr);
  }

  inline bool is_head_block(block const& b) {
    return b.is_head_block; //heap_blocks.count(b.ptr) > 0;
  }

  inline void find_and_free_block(void *p, size_t size, cudaStream_t stream)
  {
    raii_timer(2);

    if (p == nullptr) return;

    auto i = std::find_if(allocated_blocks.begin(), allocated_blocks.end(),
                          [p](block const& b) { return b.ptr == p; });
    
    if (i != allocated_blocks.end()) { // found
      //assert(i->size == detail::round_up_safe(size, allocation_alignment));
      sync_blocks[stream].insert_and_merge(*i);
      allocated_blocks.erase(i);
    }
    else {
      throw std::runtime_error("Pointer not allocated by this resource.");
    }
  }

  inline block block_from_heap(size_t size, cudaStream_t stream)
  {
    void* p = heap_resource->allocate(size, stream);
    if (p == nullptr)
      throw std::bad_alloc{};
    block b{reinterpret_cast<char*>(p), size, true};
    heap_blocks[b.ptr] = b;
    return b;
  }

  inline void free_all()
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
    bytes = detail::round_up_safe(bytes, allocation_alignment);
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
  std::pair<size_t,size_t> do_get_mem_info( cudaStream_t stream) const {
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

  // allocated_blocks: map of allocated [ptr, block] pairs
  std::list<block> allocated_blocks;

  // blocks allocated from heap: so they can be easily freed
  std::unordered_map<char*, block> heap_blocks;
};

}  // namespace mr
}  // namespace rmm
