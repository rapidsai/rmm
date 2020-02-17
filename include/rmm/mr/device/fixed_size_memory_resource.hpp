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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime_api.h>

#include <list>
#include <unordered_map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {
/**---------------------------------------------------------------------------*
 * @brief Allocates fixed-size memory blocks.
 * 
 * Supports only allocations of size smaller than the configured block_size.
 *---------------------------------------------------------------------------**/
class fixed_size_memory_resource : public device_memory_resource {
 public:

  
  static constexpr std::size_t default_block_size = 1<<20; // 1 MiB
  static constexpr std::size_t default_heap_chunk_size = 128 * default_block_size;
  static constexpr std::size_t allocation_alignment = 256;

  explicit fixed_size_memory_resource(std::size_t block_size = default_block_size,
                                      std::size_t heap_chunk_size = default_heap_chunk_size,
                                      rmm::mr::device_memory_resource *heap_resource =
                                        rmm::mr::get_default_resource()) 
  : heap_resource_{heap_resource} {
    block_size_ = rmm::detail::align_up(block_size, allocation_alignment); 
    heap_chunk_size_ = rmm::detail::align_up(heap_chunk_size, block_size);

    // allocate initial blocks and insert into free list
    new_blocks_from_heap(0);
  }

  virtual ~fixed_size_memory_resource() {
    free_all();
  }

  /**---------------------------------------------------------------------------*
   * @brief Queries whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   *---------------------------------------------------------------------------**/
  bool supports_streams() const noexcept override { return true; }

  std::size_t get_block_size() { return block_size_; }

 private:

  using free_list = std::list<void*>;

  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws std::bad_alloc if size > block_size (constructor parameter)
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    bytes = rmm::detail::align_up(bytes, allocation_alignment);
    if (bytes > block_size_)
      throw std::bad_alloc();
    
    return get_block(stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    assert(bytes < block_size_);
    sync_blocks_[stream].push_back(p);
  }

  /**---------------------------------------------------------------------------*
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   *---------------------------------------------------------------------------**/
  std::pair<std::size_t, std::size_t> do_get_mem_info( cudaStream_t stream) const override {
    return std::make_pair(0, 0);  
  }

  void* block_from_sync_list(cudaStream_t stream) {
    free_list& blocks = sync_blocks_.at(stream);
    void* p = nullptr;
    if (blocks.size() > 0) {
       p = blocks.front();
       blocks.pop_front();
    }
    // insert all remaining blocks from sync list into no-sync list
    if (p != nullptr) { // found one
      cudaError_t result = cudaStreamSynchronize(stream);

      if (result != cudaErrorInvalidResourceHandle && // stream deleted
          result != cudaSuccess)                      // stream synced
        throw std::runtime_error{"cudaStreamSynchronize failure"};

      // insert all other blocks into the no_sync list
      no_sync_blocks_.insert(no_sync_blocks_.end(), blocks.cbegin(), blocks.cend());
      blocks.clear();

      // remove this stream from the freelist
      sync_blocks_.erase(stream);
    }
    return p;
  }

  void* get_block(cudaStream_t stream) {
    // Try to find a larger block that doesn't require syncing
    void* p = nullptr;
    if (no_sync_blocks_.size() > 0) {
       p = no_sync_blocks_.front();
       no_sync_blocks_.pop_front();
    }

    // nothing in no-sync free list, look for one on a stream
    if (p == nullptr) {

      // Try to find a larger block in a different stream
      for (auto s : sync_blocks_) {
        if (s.first != stream) p = block_from_sync_list(s.first);
      }

      // nothing available in other streams, look in current stream's list
      if (p == nullptr && (sync_blocks_.find(stream) != sync_blocks_.end())) {
          p = block_from_sync_list(stream);
      }

      // no blocks available, so create more
      if (p == nullptr) {
        new_blocks_from_heap(stream);
        p = get_block(stream);
      }
    }

    return p;
  }

  void new_blocks_from_heap(cudaStream_t stream) {
    void* p = heap_resource_->allocate(heap_chunk_size_, stream);
    if (!p) throw std::bad_alloc();
    heap_blocks_[p] = heap_chunk_size_;

    auto num_blocks = heap_chunk_size_ / block_size_;

    auto g = [p, this](int i) { return static_cast<char*>(p) + i * block_size_; };

    auto first = thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), g); 
    auto last  = thrust::make_transform_iterator(thrust::make_counting_iterator(num_blocks), g); 
    no_sync_blocks_.insert(no_sync_blocks_.cend(), first, last);
  }

  void free_all()
  {
    for (auto b : heap_blocks_) heap_resource_->deallocate(b.first, b.second);
    heap_blocks_.clear();
  }

  device_memory_resource *heap_resource_;

  std::size_t block_size_;      // size of blocks this MR allocates
  std::size_t heap_chunk_size_; // size of chunks allocated from heap MR

  // no-sync free list: list of unencumbered blocks
  // no need to sync a stream to allocate from this list
  free_list no_sync_blocks_;

  // sync free lists: map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list
  std::unordered_map<cudaStream_t, free_list> sync_blocks_;

  // blocks allocated from heap: so they can be easily freed
  // blocks are ptr/size pairs
  std::unordered_map<void*, size_t> heap_blocks_;
};
}  // namespace mr
}  // namespace rmm
