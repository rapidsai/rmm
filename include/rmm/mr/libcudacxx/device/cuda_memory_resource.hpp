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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>

#include <cuda/memory_resource>

#include <cassert>

namespace rmm {
namespace mr {

/**
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 */
class cuda_memory_resource final
  : public cuda::stream_ordered_memory_resource<cuda::memory_kind::device> {
 public:
  cuda_memory_resource()                            = default;
  ~cuda_memory_resource()                           = default;
  cuda_memory_resource(cuda_memory_resource const&) = default;
  cuda_memory_resource(cuda_memory_resource&&)      = default;
  cuda_memory_resource& operator=(cuda_memory_resource const&) = default;
  cuda_memory_resource& operator=(cuda_memory_resource&&) = default;

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment. Alignments greater than
   * this are not supported.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   * @throws `rmm::logic_error` if the requested alignment is greater than 256B
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    RMM_EXPECTS(detail::is_aligned(detail::CUDA_ALLOCATION_ALIGNMENT, alignment),
                "Unsupported alignment");
    void* p{nullptr};
    RMM_CUDA_TRY(cudaMalloc(&p, bytes), rmm::bad_alloc);
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * Alignments greater than 256B are not supported and behavior is undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t, std::size_t alignment) override
  {
    assert(detail::is_aligned(detail::CUDA_ALLOCATION_ALIGNMENT, alignment));
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(p));
  }

  void* do_allocate_async(std::size_t bytes, std::size_t alignment, cuda::stream_view) override
  {
    return do_allocate(bytes, alignment);
  }

  void do_deallocate_async(void* p,
                           std::size_t bytes,
                           std::size_t alignment,
                           cuda::stream_view) override
  {
    return do_deallocate(p, bytes, alignment);
  };

  /**
   * @brief Compare this resource to another.
   *
   * Two cuda_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(memory_resource const& other) const noexcept override
  {
    return dynamic_cast<cuda_memory_resource const*>(&other) != nullptr;
  }
};
}  // namespace mr
}  // namespace rmm
