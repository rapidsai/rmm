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

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/detail/free_list.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <list>

namespace rmm {
namespace mr {
namespace detail {

struct event_block : public block {
  event_block() = default;
  event_block(char* ptr, size_t size, bool is_head) : block(ptr, size, is_head) {}

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param b block to merge
   * @return block The merged block
   */
  event_block merge(event_block&& b) noexcept
  {
    assert(is_contiguous_before(b));
    size_bytes += b.size();
    events.splice(events.end(), std::move(b.events));
    b.ptr        = nullptr;
    b.size_bytes = 0;
    return *this;
  }

  void record(cudaStream_t stream) noexcept
  {
    cudaEvent_t event = nullptr;
    assert(cudaSuccess == cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    assert(cudaSuccess == cudaEventRecord(event, stream));

    events.push_back(event);
  }

 protected:
  std::list<cudaEvent_t> events;  ///< List of events to wait on before allocating from this block
};

};  // namespace detail
};  // namespace mr
};  // namespace rmm