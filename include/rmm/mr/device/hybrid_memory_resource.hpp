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

/**---------------------------------------------------------------------------*
 * @brief Allocates fixed-size memory blocks.
 * 
 * Supports only allocations of size smaller than the configured block_size.
 *---------------------------------------------------------------------------**/
 template <typename SmallAllocMemoryResource, typename LargeAllocMemoryResource>
class hybrid_memory_resource : public device_memory_resource {
 public:

  static constexpr std::size_t default_max_small_size = 1 << 22; // 4 MiB

  explicit hybrid_memory_resource(SmallAllocMemoryResource *small_mr,
                                  LargeAllocMemoryResource *large_mr,
                                  std::size_t max_small_size = default_max_small_size) 
  : small_mr_{small_mr}, large_mr_{large_mr}, max_small_size_{max_small_size} {}

  virtual ~hybrid_memory_resource() = default;

  /**---------------------------------------------------------------------------*
   * @brief Queries whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   *---------------------------------------------------------------------------**/
  bool supports_streams() const noexcept override { return true; }

  SmallAllocMemoryResource* get_small_mr() { return small_mr_; }
  LargeAllocMemoryResource* get_large_mr() { return large_mr_; }

 private:

  rmm::mr::device_memory_resource& get_allocator(std::size_t bytes) {
    if (bytes <= max_small_size_) return *small_mr_;
    else return *large_mr_;
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

  std::size_t max_small_size_; // power of 2 maximum fixed size allocator

  SmallAllocMemoryResource *small_mr_; // allocator for <= max_small_size
  LargeAllocMemoryResource *large_mr_; // allocator for > max_small_size
};
}  // namespace mr
}  // namespace rmm
