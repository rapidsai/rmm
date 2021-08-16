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
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

namespace rmm {
namespace mr {

/**
 * @brief A device memory resource that simulates a fix-sized GPU.
 *
 * Only allocation calls are simulated. New memory is allocated sequentially in monotonically
 * increasing address based on the requested size, until the predetermined size is exceeded.
 *
 * Deallocation calls are ignored.
 */
class simulated_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Construct a `simulated_memory_resource`.
   *
   * @param memory_size_bytes The size of the memory to simulate.
   */
  explicit simulated_memory_resource(std::size_t memory_size_bytes)
    : begin_{reinterpret_cast<char*>(0x100)},
      end_{reinterpret_cast<char*>(begin_ + memory_size_bytes)}
  {
  }

  // Disable copy (and move) semantics.
  simulated_memory_resource(simulated_memory_resource const&) = delete;
  simulated_memory_resource& operator=(simulated_memory_resource const&) = delete;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool false
   */
  bool supports_streams() const noexcept override { return false; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return false
   */
  bool supports_get_mem_info() const noexcept override { return false; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @note Stream argument is ignored
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view) override
  {
    RMM_EXPECTS(begin_ + bytes <= end_, rmm::bad_alloc, "Simulated memory size exceeded");
    auto p = static_cast<void*>(begin_);
    begin_ += bytes;
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note This call is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate_async(void* p, std::size_t, std::size_t, cuda_stream_view) override {}

  /**
   * @brief Get free and available memory for memory resource.
   *
   * @param stream to execute on.
   * @return std::pair containing free_size and total_size of memory.
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    return std::make_pair(0, 0);
  }

 private:
  char* begin_;
  char* end_;
};
}  // namespace mr
}  // namespace rmm
