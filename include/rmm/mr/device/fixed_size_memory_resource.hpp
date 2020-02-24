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

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/detail/error.hpp>

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
/**
 * @brief Allocates fixed-size memory blocks.
 * 
 * Supports only allocations of size smaller than the configured block_size.
 */
template <typename UpstreamResource>
class fixed_size_memory_resource : public device_memory_resource {
 public:

  // A block is the fixed size this resource alloates
  static constexpr std::size_t default_block_size = 1<<20; // 1 MiB
  // This is the number of blocks that the pool starts out with, and also the number of 
  // blocks by which the pool grows when all of its current blocks are allocated
  static constexpr std::size_t default_blocks_to_preallocate = 128;
  // The required alignment of this allocator
  static constexpr std::size_t allocation_alignment = 256;

  explicit fixed_size_memory_resource(UpstreamResource *upstream_resource,
                                      std::size_t block_size = default_block_size,
                                      std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
  : upstream_resource_{upstream_resource} {
    block_size_ = rmm::detail::align_up(block_size, allocation_alignment); 
    upstream_chunk_size_ = block_size * default_blocks_to_preallocate;

    // allocate initial blocks and insert into free list
    new_blocks_from_upstream(0);
  }

  virtual ~fixed_size_memory_resource() {
    free_all();
  }

  /**
   * @brief Queries whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   */
  bool supports_streams() const noexcept override { return true; }

  std::size_t get_block_size() const noexcept { return block_size_; }

 private:

  using free_list = std::list<void*>;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc if `bytes` > `block_size` (constructor parameter)
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    bytes = rmm::detail::align_up(bytes, allocation_alignment);
    RMM_EXPECTS(bytes <= block_size_, rmm::bad_alloc, "bytes must be <= block_size");

    return get_block(stream);
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws rmm::bad_alloc if `bytes` > `block_size` (constructor parameter)
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    bytes = rmm::detail::align_up(bytes, allocation_alignment);
    RMM_EXPECTS(bytes <= block_size_, rmm::bad_alloc, "bytes must be <= block_size");

    sync_blocks_[stream].push_back(p);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info( cudaStream_t stream) const override {
    return std::make_pair(0, 0);  
  }

  // return a block from the free list associated with the specified stream, and move all other 
  // blocks in the list to the no-sync free list.
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

  // Returns a pointer to a free block. Attempts to return a block that is not associated with
  // a stream first, and if there are none, returns a block from a sync free list. If no blocks 
  // are available, allocates from the upstream heap resource
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
        new_blocks_from_upstream(stream);
        p = get_block(stream);
      }
    }

    return p;
  }

  // Allocate new blocks from the upstream memory resource into the free list
  void new_blocks_from_upstream(cudaStream_t stream) {
    void* p = upstream_resource_->allocate(upstream_chunk_size_, stream);
    upstream_blocks_[p] = upstream_chunk_size_;

    auto num_blocks = upstream_chunk_size_ / block_size_;

    auto g = [p, this](int i) { return static_cast<char*>(p) + i * block_size_; };

    auto first = thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), g); 
    auto last  = thrust::make_transform_iterator(thrust::make_counting_iterator(num_blocks), g); 
    no_sync_blocks_.insert(no_sync_blocks_.cend(), first, last);
  }

  // free all allocated memory
  void free_all()
  {
    for (auto b : upstream_blocks_) upstream_resource_->deallocate(b.first, b.second);
    upstream_blocks_.clear();
    no_sync_blocks_.clear();
    sync_blocks_.clear();
  }

  UpstreamResource *upstream_resource_; // The resource from which to allocate new blocks

  std::size_t block_size_;      // size of blocks this MR allocates
  std::size_t upstream_chunk_size_; // size of chunks allocated from heap MR

  // TODO I tried removing this and just using the stream_blocks as in sub_memory_resource
  // but it was much slower, presumably because of the lookup overhead on every alloc, which is 
  // otherwise very cheap.
  // no-sync free list: list of unencumbered blocks
  // no need to sync a stream to allocate from this list
  free_list no_sync_blocks_;

  // sync free lists: map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list
  std::unordered_map<cudaStream_t, free_list> sync_blocks_;

  // blocks allocated from heap: so they can be easily freed
  // blocks are ptr/size pairs
  std::unordered_map<void*, size_t> upstream_blocks_;
};
}  // namespace mr
}  // namespace rmm
