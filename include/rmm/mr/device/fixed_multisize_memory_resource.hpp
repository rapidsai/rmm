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
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/aligned.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime_api.h>

#include <list>
#include <unordered_map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>
#include <memory>

#include <iostream>

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {

/**
 * @brief Allocates fixed-size memory blocks of a range of sizes.
 * 
 * Allocates blocks in the range `[min_size], max_size]` in power of two steps, 
 * where `min_size` and `max_size` are both powers of two.
 * 
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
 template <typename UpstreamResource>
class fixed_multisize_memory_resource : public device_memory_resource {
 public:

  static constexpr std::size_t default_min_size = 1 << 18; // 256 KiB
  static constexpr std::size_t default_max_size = 1 << 22; // 4 MiB

  /**
   * @brief Construct a new fixed multisize memory resource object
   * 
   * @throws rmm::logic_error if min_size or max_size is not a power of two.
   *
   * @param upstream_resource The upstream memory resource used to allocate pools of blocks
   * @param min_size The minimum fixed block size to allocate
   * @param max_size The maximum fixed block size to allocate
   * @param initial_blocks_per_size The number of blocks to preallocate from the upstream memory
   *        resource
   */
  explicit fixed_multisize_memory_resource(UpstreamResource *upstream_resource,
                                           std::size_t min_size = default_min_size,
                                           std::size_t max_size = default_max_size,
                                           std::size_t initial_blocks_per_size = 128)
  : upstream_mr_{upstream_resource}, min_size_{min_size}, max_size_{max_size} {
    RMM_EXPECTS(rmm::detail::is_pow2(min_size), "Minimum size must be a power of two");
    RMM_EXPECTS(rmm::detail::is_pow2(max_size), "Maximum size must be a power of two");
    
    // allocate initial blocks and insert into free list
    for (std::size_t i = min_size; i <= max_size; i *= 2) {
      fixed_size_mr_.emplace_back(
        new fixed_size_memory_resource<UpstreamResource>(upstream_resource, i, 
                                                         initial_blocks_per_size * i)
      );
    }
  }

  /**
   * @brief Destroy the fixed_multisize_memory_resource and free all memory allocated from the
   *        upstream resource.
   */
  virtual ~fixed_multisize_memory_resource() {
    fixed_size_mr_.clear(); // must be deleted first since fixed_size_mrs use large_size_mr_
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
  bool supports_get_mem_info() const noexcept override {return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  UpstreamResource* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Get the minimum block size that this memory_resource can allocate.
   * 
   * @return std::size_t The minimum block size this memory_resource can allocate.
   */
  std::size_t get_min_size() const noexcept { return min_size_; }

  /**
   * @brief Get the maximum block size that this memory_resource can allocate.
   * 
   * @return std::size_t The maximum block size this memory_resource can allocate.
   */
  std::size_t get_max_size() const noexcept { return max_size_; }

 private:

  /**
   * @brief Get the memory resource for the requested size
   *
   * Chooses a memory_resource that allocates the smallest blocks larger than `bytes`.
   *
   * @throws rmm::bad_alloc if the size is larger than the maximum block size.
   *
   * @param bytes Requested allocation size in bytes
   * @return rmm::mr::device_memory_resource& memory_resource that can allocate the requested size.
   */
  device_memory_resource& get_resource(std::size_t bytes) {
    if (bytes <= max_size_) {
      auto iter = std::find_if(fixed_size_mr_.begin(), fixed_size_mr_.end(),
                   [bytes](std::unique_ptr<fixed_size_memory_resource<UpstreamResource>>& res) { 
                     return bytes <= res.get()->get_block_size();
                    });
      if (iter != fixed_size_mr_.end())
        return *iter->get();
    }

    RMM_FAIL("bytes must be <= max_size", rmm::bad_alloc);
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc if size > block_size (constructor parameter)
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    return get_resource(bytes).allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws std::bad_alloc if size > block_size (constructor parameter)
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    get_resource(bytes).deallocate(p, bytes, stream);
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

  UpstreamResource *upstream_mr_; // The upstream memory_resource from which to allocate blocks.

  std::size_t min_size_; // size of smallest blocks allocated
  std::size_t max_size_; // size of largest blocks allocated

  // allocators for fixed-size blocks <= max_fixed_size_
  std::vector<std::unique_ptr<fixed_size_memory_resource<UpstreamResource>>> fixed_size_mr_;
};
}  // namespace mr
}  // namespace rmm
