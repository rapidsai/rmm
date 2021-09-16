/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

enum cuda_event_flags {
  /** Default event flag. */
  EVENT_DEFAULT = cudaEventDefault,
  /** Event uses blocking synchronization. */
  EVENT_BLOCKING_SYNC = cudaEventBlockingSync,
  /** Event will not record timing data. */
  EVENT_DISABLE_TIMING = cudaEventDisableTiming,
  /** Event is suitable for interprocess use. cudaEventDisableTiming must be set. */
  EVENT_INTERPROCESS = cudaEventInterprocess
};

enum cuda_event_record_flags {
  /** Default event creation flag. */
  EVENT_RECORD_DEFAULT = cudaEventRecordDefault,
  /** Event is captured in the graph as an external event node when performing stream capture. */
  EVENT_RECORD_EXTERNAL = cudaEventRecordExternal
};

enum cuda_event_wait_flags {
  /** Default event creation flag. */
  EVENT_WAIT_DEFAULT = cudaEventWaitDefault,
  /** Event is captured in the graph as an external event node when performing stream capture. */
  EVENT_WAIT_EXTERNAL = cudaEventWaitExternal
};

/**
 * @brief Strongly-typed non-owning wrapper for CUDA events.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the event it wraps.
 */
class cuda_event_view {
 public:
  constexpr cuda_event_view()                       = delete;
  constexpr cuda_event_view(cuda_event_view const&) = default;
  constexpr cuda_event_view(cuda_event_view&&)      = default;
  constexpr cuda_event_view& operator=(cuda_event_view const&) = default;
  constexpr cuda_event_view& operator=(cuda_event_view&&) = default;
  ~cuda_event_view()                                      = default;

  /**
   * @brief Implicit conversion from cudaEvent_t.
   */
  constexpr cuda_event_view(cudaEvent_t event) noexcept : event_{event} {}

  /**
   * @brief Get the wrapped event.
   *
   * @return cudaEvent_t The wrapped event.
   */
  constexpr cudaEvent_t value() const noexcept { return event_; }

  /**
   * @brief Implicit conversion to cudaEvent_t.
   */
  constexpr operator cudaEvent_t() const noexcept { return value(); }

  /**
   * @brief Computes the elapsed time between since the given event till the viewed event.
   *
   * @return time in milliseconds.
   */
  float elapsed_time_since(cuda_event_view prev_event) const
  {
    float ms;
    RMM_CUDA_TRY(cudaEventElapsedTime(&ms, prev_event.value(), value()));
    return ms;
  }

  /**
   * @brief Queries the viewed event's status.
   *
   * @return whether all captured work has been completed.
   */
  bool query_status() const
  {
    auto status = cudaEventQuery(value());
    if (status == cudaErrorNotReady) return false;
    RMM_CUDA_TRY(status);
    return true;
  }

  /**
   * @brief Synchronize the viewed CUDA event (block host thread till completion).
   *
   * Calls `cudaEventSynchronize()`.
   *
   * @throw rmm::cuda_error if event synchronization fails
   */
  void wait() const { RMM_CUDA_TRY(cudaEventSynchronize(event_)); }

  /**
   * @brief Synchronize the viewed CUDA event (block host thread till completion).
   *
   * Calls `cudaEventSynchronize()` asserting the CUDA_SUCCESS result
   */
  void wait_no_throw() const noexcept { RMM_ASSERT_CUDA_SUCCESS(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_;
};

/**
 * @brief Equality comparison operator for events
 *
 * @param lhs The first event view to compare
 * @param rhs The second event view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(cuda_event_view lhs, cuda_event_view rhs)
{
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for events
 *
 * @param lhs The first event view to compare
 * @param rhs The second event view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(cuda_event_view lhs, cuda_event_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output event operator for printing / logging events
 *
 * @param os The output ostream
 * @param sv The cuda_event_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_event_view sv)
{
  os << sv.value();
  return os;
}

}  // namespace rmm
