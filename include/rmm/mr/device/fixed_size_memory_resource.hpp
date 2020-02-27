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
#include <rmm/detail/error.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime_api.h>

#include <list>
#include <map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>

namespace rmm {

namespace mr {

/**
 * @brief A `device_memory_resource` which allocates memory blocks of a single fixed size.
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

  /**
   * @brief Construct a new `fixed_size_memory_resource` that allocates memory from 
   * `upstream_resource`.
   *
   * When the pool of blocks is all allocated, grows the pool by allocating
   * `blocks_to_preallocate` more blocks from `upstream_mr`.
   * 
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  explicit fixed_size_memory_resource(
      UpstreamResource* upstream_mr,
      std::size_t block_size = default_block_size,
      std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
      : upstream_mr_{upstream_mr} {
    block_size_ = rmm::detail::align_up(block_size, allocation_alignment); 
    upstream_chunk_size_ = block_size * default_blocks_to_preallocate;

    // allocate initial blocks and insert into free list
    new_blocks_from_upstream(0, stream_blocks_[0]);
  }

  /**
   * @brief Destroy the `fixed_size_memory_resource` and free all memory allocated from upstream.
   * 
   */
  virtual ~fixed_size_memory_resource() {
    free_all();
  }

  /**
   * @brief Query whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   * 
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  UpstreamResource* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Get the size of blocks allocated by this memory resource. 
   * 
   * @return std::size_t size in bytes of allocated blocks.
   */
  std::size_t get_block_size() const noexcept { return block_size_; }

 private:

  // blocks are maintained in a simple list of pointers
  // Allocation is simply popping off the list, and freeing is pushing onto the back.
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

    stream_blocks_[stream].push_back(p);
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

  /**
   * @brief Find a free block in `free_list` `blocks`.
   * 
   * @param blocks The `free_list` to look in for a free block.
   * @param list_stream The stream that all blocks in `blocks` were associated with when they were
   *                    freed.
   * @param stream The stream on which the allocation is being requested.
   * @return block A pointer to memory of `get_block_size()` bytes, or nullptr if no blocks are
   *               available in `blocks`.
   */
  void* block_from_sync_list(free_list &blocks, cudaStream_t list_stream, cudaStream_t stream) {
    void* p = nullptr;

    if (!blocks.empty()) {
      p = blocks.front();
      blocks.pop_front();
    }

    // If we found a block associated with a different stream, 
    // we have to synchronize the stream in order to use it
    if ((list_stream != stream) && (p != nullptr)) {
      cudaError_t result = cudaStreamSynchronize(list_stream);

      if (result != cudaErrorInvalidResourceHandle && // stream deleted
          result != cudaSuccess)                      // stream synced
        throw std::runtime_error{"cudaStreamSynchronize failure"};

      // insert all other blocks into this stream's list
      // TODO: Should we do this? This could cause thrashing between two 
      // streams. For future analysis.
      auto blocks_for_stream = stream_blocks_[stream];
      blocks_for_stream.splice(blocks_for_stream.end(), blocks);

      // remove this stream from the freelist
      stream_blocks_.erase(list_stream);
    }

    return p;
  }

  /**
   * @brief Find an available block in the pool, for use on `stream`.
   * 
   * Attempts to find a free block that was last used on `stream` to avoid synchronization. If none
   * is available, it finds a block last used on another stream. In this case, the stream associated
   * with the found block is synchronized to ensure all asynchronous work on the memory is finished
   * before it is used on `stream`.
   * 
   * @param stream The stream on which the allocation will be used.
   * @return block A pointer to memory of size `get_block_size()`.
   */
  void* get_block(cudaStream_t stream) {
    void* p = nullptr;

    // Try to find a block in the same stream
    auto iter = stream_blocks_.find(stream);
    if (iter != stream_blocks_.end())
      p = block_from_sync_list(iter->second, stream, stream);

    // nothing in this stream's free list, look for one on another stream
    if (p == nullptr) {
      // Try to find a larger block in a different stream
      auto s = stream_blocks_.begin();
      while (p == nullptr && s != stream_blocks_.end()) {
        if (s->first != stream) p = block_from_sync_list(s->second, s->first, stream);
        s++;
      }
    }
    
    // nothing available in other streams, get new blocks
    if (p == nullptr) {
      free_list* plist = nullptr; // avoid searching for this stream's list again
      if (iter != stream_blocks_.end()) {
        plist = &iter->second;
      }

      if (plist == nullptr)
        plist = &stream_blocks_[stream];

      new_blocks_from_upstream(stream, *plist);
      p = get_block(stream);
    }

    return p;
  }

  // 
  /**
   * @brief Allocate new blocks from the upstream memory resource into `free list` `blocks`.
   * 
   * @param stream The stream to associate the new blocks with.
   * @param blocks The `free_list` to insert the blocks into.
   */
  void new_blocks_from_upstream(cudaStream_t stream, free_list& blocks) {
    void* p = upstream_mr_->allocate(upstream_chunk_size_, stream);
    upstream_blocks_[p] = upstream_chunk_size_;

    auto num_blocks = upstream_chunk_size_ / block_size_;

    auto g = [p, this](int i) { return static_cast<char*>(p) + i * block_size_; };

    auto first = thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), g); 
    auto last  = thrust::make_transform_iterator(thrust::make_counting_iterator(num_blocks), g); 
    blocks.insert(blocks.cend(), first, last);
  }

  /**
   * @brief free all memory allocated using the upstream resource.
   * 
   */
  void free_all()
  {
    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.first, b.second);
    upstream_blocks_.clear();
    stream_blocks_.clear();
  }

  UpstreamResource* upstream_mr_;  // The resource from which to allocate new blocks

  std::size_t block_size_;          // size of blocks this MR allocates
  std::size_t upstream_chunk_size_; // size of chunks allocated from heap MR

  // stream free lists: map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list
  std::map<cudaStream_t, free_list> stream_blocks_;

  // blocks allocated from heap: so they can be easily freed
  // blocks are ptr/size pairs
  std::map<void*, size_t> upstream_blocks_;
};
}  // namespace mr
}  // namespace rmm
