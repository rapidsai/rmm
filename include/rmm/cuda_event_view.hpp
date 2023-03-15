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
#include <limits>
#include <type_traits>

namespace rmm {

enum cuda_event_flags : unsigned int {
  /** Default event flag. */
  EVENT_DEFAULT = cudaEventDefault,
  /** Event uses blocking synchronization. */
  EVENT_BLOCKING_SYNC = cudaEventBlockingSync,
  /** Event will not record timing data. */
  EVENT_DISABLE_TIMING = cudaEventDisableTiming,
  /** Event is suitable for interprocess use. cudaEventDisableTiming must be set. */
  EVENT_INTERPROCESS = cudaEventInterprocess
};

constexpr inline cuda_event_flags operator|(cuda_event_flags a, cuda_event_flags b)
{
  return static_cast<cuda_event_flags>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

enum cuda_event_record_flags : unsigned int {
/** Default event creation flag. */
#if CUDART_VERSION < 11010
  EVENT_RECORD_DEFAULT = 0,
#else
  EVENT_RECORD_DEFAULT = cudaEventRecordDefault,
  /** Event is captured in the graph as an external event node when performing stream capture. */
  EVENT_RECORD_EXTERNAL = cudaEventRecordExternal
#endif
};

constexpr inline cuda_event_record_flags operator|(cuda_event_record_flags a,
                                                   cuda_event_record_flags b)
{
  return static_cast<cuda_event_record_flags>(static_cast<unsigned int>(a) |
                                              static_cast<unsigned int>(b));
}

enum cuda_event_wait_flags : unsigned int {
/** Default event creation flag. */
#if CUDART_VERSION < 11010
  EVENT_WAIT_DEFAULT = 0,
#else
  EVENT_WAIT_DEFAULT = cudaEventWaitDefault,
  /** Event is captured in the graph as an external event node when performing stream capture. */
  EVENT_WAIT_EXTERNAL = cudaEventWaitExternal
#endif
};

constexpr inline cuda_event_wait_flags operator|(cuda_event_wait_flags a, cuda_event_wait_flags b)
{
  return static_cast<cuda_event_wait_flags>(static_cast<unsigned int>(a) |
                                            static_cast<unsigned int>(b));
}

/** @brief An event view with flags provided at runtime. */
class cuda_event_view_ {
 public:
  constexpr cuda_event_view_()                        = delete;
  constexpr cuda_event_view_(cuda_event_view_ const&) = default;
  constexpr cuda_event_view_(cuda_event_view_&&)      = default;
  constexpr cuda_event_view_& operator=(cuda_event_view_ const&) = default;
  constexpr cuda_event_view_& operator=(cuda_event_view_&&) = default;
  ~cuda_event_view_()                                       = default;

  /**
   * @brief Implicit conversion from cudaEvent_t.
   */
  constexpr cuda_event_view_(cudaEvent_t event) noexcept : event_{event} {}

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
  float elapsed_time_since(const cuda_event_view_& prev_event) const
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

template <cuda_event_flags Flags>
class cuda_event;

/**
 * @brief Strongly-typed non-owning wrapper for CUDA events.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the event it wraps.
 */
template <cuda_event_flags Flags = EVENT_DEFAULT>
class cuda_event_view : public cuda_event_view_ {
 public:
  constexpr cuda_event_view()                              = delete;
  constexpr cuda_event_view(cuda_event_view<Flags> const&) = default;
  constexpr cuda_event_view(cuda_event_view<Flags>&&)      = default;
  constexpr cuda_event_view(cuda_event<Flags> const&) noexcept;
  constexpr cuda_event_view& operator=(cuda_event_view<Flags> const&) = default;
  constexpr cuda_event_view& operator=(cuda_event_view<Flags>&&) = default;
  ~cuda_event_view()                                             = default;

  /**
   * @brief Computes the elapsed time between since the given event till the viewed event.
   *
   * @return time in milliseconds.
   */
  inline float elapsed_time_since(const cuda_event_view_& prev_event) const
  {
    static_assert((Flags & EVENT_DISABLE_TIMING) == 0,
                  "EVENT_DISABLE_TIMING must not be set for this event.");
    return cuda_event_view_::elapsed_time_since(prev_event);
  }

  /**
   * @brief Computes the elapsed time between since the given event till the viewed event.
   *
   * @return time in milliseconds.
   */
  template <cuda_event_flags FlagsPrev>
  inline float elapsed_time_since(const cuda_event_view<FlagsPrev>& prev_event) const
  {
    static_assert((FlagsPrev & EVENT_DISABLE_TIMING) == 0,
                  "EVENT_DISABLE_TIMING must not be set for the previous event.");
    return elapsed_time_since(static_cast<const cuda_event_view_&>(prev_event));
  }
};

/**
 * @brief Equality comparison operator for events
 *
 * @param lhs The first event view to compare
 * @param rhs The second event view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(cuda_event_view_ lhs, cuda_event_view_ rhs)
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
inline bool operator!=(cuda_event_view_ lhs, cuda_event_view_ rhs) { return not(lhs == rhs); }

/**
 * @brief Output event operator for printing / logging events
 *
 * @param os The output ostream
 * @param sv The cuda_event_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_event_view_ sv)
{
  os << sv.value();
  return os;
}

}  // namespace rmm
