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
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/sub_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <list>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>


// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {

namespace detail {
  inline bool is_pow2(std::size_t x) { return (x & (x - 1)) == 0; }
}

/**---------------------------------------------------------------------------*
 * @brief Allocates fixed-size memory blocks.
 * 
 * Supports only allocations of size smaller than the configured block_size.
 *---------------------------------------------------------------------------**/
class hybrid_memory_resource : public device_memory_resource {
 public:

  
  static constexpr std::size_t default_min_fixed_size = 1 << 18; // 256 KiB
  static constexpr std::size_t default_max_fixed_size = 1 << 22; // 4 MiB

  explicit hybrid_memory_resource(std::size_t min_fixed_size = default_min_fixed_size,
                                  std::size_t max_fixed_size = default_max_fixed_size,
                                  std::size_t initial_blocks_per_size = 128,
                                  rmm::mr::device_memory_resource *heap_resource =
                                    rmm::mr::get_default_resource()) 
  : max_fixed_size_{max_fixed_size} {
    if (!detail::is_pow2(min_fixed_size) || !detail::is_pow2(max_fixed_size)) 
      throw std::logic_error("Only power of two fixed sizes supported");
    
    large_size_mr_ = new sub_memory_resource();

    // allocate initial blocks and insert into free list
    for (std::size_t i = min_fixed_size; i <= max_fixed_size; i *= 2) {
      fixed_size_mr_.emplace_back(
        new fixed_size_memory_resource(i, initial_blocks_per_size * i, large_size_mr_)
      );
    }
  }

  virtual ~hybrid_memory_resource() {
    fixed_size_mr_.clear(); // must be deleted first since fixed_size_mrs use large_size_mr_
    delete large_size_mr_;
  }

  /**---------------------------------------------------------------------------*
   * @brief Queries whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   *---------------------------------------------------------------------------**/
  bool supports_streams() const noexcept override { return true; }

 private:

  rmm::mr::device_memory_resource& get_allocator(std::size_t bytes) {
    if (bytes > max_fixed_size_) return *large_size_mr_;
    else {
      auto iter = fixed_size_mr_.begin();
      while (bytes > iter->get()->get_block_size())
        iter = std::next(iter);
      if (iter != fixed_size_mr_.end())
        return *iter->get();
      else
        throw std::bad_alloc();
    }
  }

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
    return get_allocator(bytes).allocate(bytes, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws std::bad_alloc if size > block_size (constructor parameter)
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    get_allocator(bytes).deallocate(p, bytes, stream);
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

  std::size_t max_fixed_size_; // power of 2 maximum fixed size allocator

  sub_memory_resource *large_size_mr_; // allocator for larger than max_fixed_size

  // allocators for fixed-size blocks <= max_fixed_size_
  std::vector<std::unique_ptr<fixed_size_memory_resource>> fixed_size_mr_;
  
};
}  // namespace mr
}  // namespace rmm
