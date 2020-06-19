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
#include <rmm/mr/device/detail/block.hpp>
#include <rmm/mr/device/detail/free_list.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <list>

namespace rmm {
namespace mr {
namespace detail {

struct event_block : public block {
  event_block() = default;
  event_block(char* ptr, size_t size, bool is_head) : block(ptr, size, is_head) {}

  void print() const override
  {
    std::cout << reinterpret_cast<void*>(pointer()) << " " << size() << "B " << events.size()
              << "\n";
  }

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
    // std::cout << "splicing " << b.events.size() << " events onto " << events.size() << "
    // events\n";
    events.splice(events.end(), std::move(b.events));
    b.ptr        = nullptr;
    b.size_bytes = 0;
    return *this;
  }

  template <typename EventResource>
  void record(cudaStream_t stream, EventResource& event_resource) noexcept
  {
    cudaEvent_t event = event_resource.get_event();

    auto result = cudaEventRecord(event, stream);
    assert(cudaSuccess == result);

    events.push_back(event);
  }

  template <typename EventResource>
  void await_events(cudaStream_t stream, EventResource& event_resource)
  {
    // std::cout << "Awaiting " << events.size() << " events\n";
    for (cudaEvent_t event : events) {
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
    }
    // events.clear();
    event_resource.return_events(std::move(events));
  }

 protected:
  std::list<cudaEvent_t> events;  ///< List of events to wait on before allocating from this block
};

};  // namespace detail
};  // namespace mr
};  // namespace rmm