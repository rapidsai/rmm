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

#include <thrust/mr/disjoint_sync_pool.h>
#include <thrust/mr/new.h>
#include <thrust/system/cuda/memory_resource.h>

#include <stdexcept>

namespace rmm {
namespace mr {

static std::size_t RMM_DEFAULT_DEVICE_ALIGNMENT{256};

/**
 * @brief Memory resource that allocates/deallocates using Thrust's
 * `disjoint_synchronized_pool_resource` sub-allocator.
 */
template <typename Upstream = thrust::system::cuda::detail::
              cuda_memory_resource<cudaMalloc, cudaFree, void*>,
          typename Bookkeeper = thrust::mr::new_delete_resource>
class thrust_sync_pool final : public device_memory_resource {
 public:
  /**
   * @brief Construct a `thrust_sync_pool` memory resource.
   */
  explicit thrust_sync_pool() = default;

  bool supports_streams() const noexcept override { return false; }

 private:
  thrust::mr::disjoint_synchronized_pool_resource<Upstream, Bookkeeper> _pool;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be
   * fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t) override {
    return _pool.do_allocate(bytes, RMM_DEFAULT_DEVICE_ALIGNMENT);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t) override {
    _pool.do_deallocate(p, bytes, RMM_DEFAULT_DEVICE_ALIGNMENT);
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
