/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include <rmm/detail/format.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

namespace rmm::mr {

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
    : begin_{reinterpret_cast<char*>(0x100)},                    // NOLINT
      end_{reinterpret_cast<char*>(begin_ + memory_size_bytes)}  // NOLINT
  {
  }

  ~simulated_memory_resource() override = default;

  // Disable copy (and move) semantics.
  simulated_memory_resource(simulated_memory_resource const&)            = delete;
  simulated_memory_resource& operator=(simulated_memory_resource const&) = delete;
  simulated_memory_resource(simulated_memory_resource&&)                 = delete;
  simulated_memory_resource& operator=(simulated_memory_resource&&)      = delete;

 private:
  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @note Stream argument is ignored
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view) override
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    RMM_EXPECTS(begin_ + bytes <= end_,
                "Simulated memory size exceeded (failed to allocate " +
                  rmm::detail::format_bytes(bytes) + ")",
                rmm::bad_alloc);
    auto* ptr = static_cast<void*>(begin_);
    begin_ += bytes;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @note This call is ignored.
   *
   * @param ptr Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t, cuda_stream_view) override {}

  char* begin_{};
  char* end_{};
};
}  // namespace rmm::mr
