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

#include <rmm/detail/error.hpp>
#include "device_memory_resource.hpp"

#include <cub/util_allocator.cuh>
#include <stdexcept>

namespace rmm {
namespace mr {

/**
 * @brief Memory resource that allocates/deallocates using CUB's
 * `CachingDeviceAllocator`.
 */
class cub_memory_resource final : public device_memory_resource {
 public:
  static constexpr uint32_t NO_MAX_BIN{
      cub::CachingDeviceAllocator::INVALID_BIN};
  static constexpr std::size_t NO_BYTE_LIMIT{
      cub::CachingDeviceAllocator::INVALID_SIZE};

  /**
   * @brief Construct a `cub_memory_resource` memory resource.
   */
  explicit cub_memory_resource() = default;

  /**
   * @brief Construct a new CUB memory resource with customized bins.
   *
   * Allocations are categorized and cached by bin size.
   *
   * A new allocation request of a given size will only consider cached
   * allocations within the corresponding bin.
   *
   * Bin limits progress geometrically in accordance with the growth factor
   * `bin_growth` provided during construction. Unused device allocations within
   * a larger bin cache are not reused for allocation requests that categorize
   * to smaller bin sizes.
   *
   * Allocation requests below `(bin_growth ^ min_bin)` are  rounded up to
   * `(bin_growth ^ min_bin)`. Allocations above `(bin_growth ^ max_bin)` are
   * not rounded up to the nearest bin and are simply freed when they are
   * deallocated instead of being returned to a bin-cache.
   *
   * Example:
   * ```
   * bin_growth = 8
   * min_bin = 3
   * max_bin = 7
   * 
   * bin 0: 8^3 == 512B
   * bin 1: 8^4 == 4KB
   * bin 2: 8^5 == 32KB
   * bin 3: 8^6 == 256KB
   * bin 4: 8^7 == 2MB
   * ```
   *
   * If the total storage of cached allocations on a given device will exceed
   * `max_cached_bytes`, allocations for that device are simply freed when they
   * are deallocated instead of being returned to their bin-cache.
   *
   * @param bin_growth Geometric growth factor
   * @param min_bin Minimum bin exponent (size == bin_growth ^ min_bin).
   * Defaults to 1.
   * @param max_bin Maximum bin exponent (size == bin_growth ^ max_bin).
   * Defaults to no maximum bin size.
   */
  cub_memory_resource(uint32_t bin_growth, uint32_t min_bin = 1,
                      uint32_t max_bin = NO_MAX_BIN,
                      std::size_t max_cached_bytes = NO_BYTE_LIMIT)
      : _allocator{bin_growth, min_bin, max_bin, max_cached_bytes} {}

  bool supports_streams() const noexcept override { return true; }

 private:
  cub::CachingDeviceAllocator _allocator{};

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be
   * fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    void* p{};
    RMM_CUDA_TRY(_allocator.DeviceAllocate(&p, bytes, stream), rmm::bad_alloc);
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    auto const status = _allocator.DeviceFree(&p);
    assert(cudaSuccess == status);
  }

  /**
   * @brief Unsupported.
   *
   * @throws `std::runtime_error` always.
   *
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t) const override {
    throw std::runtime_error{"Meminfo unsupported."};
  }
};
}  // namespace mr
}  // namespace rmm