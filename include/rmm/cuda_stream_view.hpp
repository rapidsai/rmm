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

#include <rmm/cuda_event_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

enum cuda_stream_flags {
  /** Default stream flag. */
  STREAM_DEFAULT = cudaStreamDefault,
  /** Stream does not synchronize with stream 0 (the NULL stream). */
  STREAM_NON_BLOCKING = cudaStreamNonBlocking
};

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class cuda_stream_view {
 public:
  constexpr cuda_stream_view()                        = default;
  constexpr cuda_stream_view(cuda_stream_view const&) = default;
  constexpr cuda_stream_view(cuda_stream_view&&)      = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view&&) = default;
  ~cuda_stream_view()                                       = default;

  // Disable construction from literal 0
  constexpr cuda_stream_view(int)            = delete;  //< Prevent cast from 0
  constexpr cuda_stream_view(std::nullptr_t) = delete;  //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from cudaStream_t.
   */
  constexpr cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

  /**
   * @brief Get the wrapped stream.
   *
   * @return cudaStream_t The wrapped stream.
   */
  constexpr cudaStream_t value() const noexcept { return stream_; }

  /**
   * @brief Implicit conversion to cudaStream_t.
   */
  constexpr operator cudaStream_t() const noexcept { return value(); }

  /**
   * @brief Return true if the wrapped stream is the CUDA per-thread default stream.
   */
  bool is_per_thread_default() const noexcept
  {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return value() == cudaStreamPerThread || value() == 0;
#else
    return value() == cudaStreamPerThread;
#endif
  }

  /**
   * @brief Return true if the wrapped stream is explicitly the CUDA legacy default stream.
   */
  bool is_default() const noexcept
  {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return value() == cudaStreamLegacy;
#else
    return value() == cudaStreamLegacy || value() == 0;
#endif
  }

  /**
   *  @brief Tells if the viewed CUDA stream is implicitly synchronized with the given stream.
   *
   *  This can happen e.g.
   *   if the two views point to the same stream
   *   or sometimes when one of them is the legacy default stream.
   */
  bool is_implicitly_synchronized(cuda_stream_view other) const
  {
    // any stream is "synchronized" with itself
    if (value() == other.value()) return true;
    // legacy + blocking streams
    unsigned int flags = 0;
    if (is_default()) {
      RMM_CUDA_TRY(cudaStreamGetFlags(other.value(), &flags));
      if ((flags & rmm::STREAM_NON_BLOCKING) == 0) return true;
    }
    if (other.is_default()) {
      RMM_CUDA_TRY(cudaStreamGetFlags(value(), &flags));
      if ((flags & rmm::STREAM_NON_BLOCKING) == 0) return true;
    }
    return false;
  }

  /**
   * @brief Synchronize the viewed CUDA stream.
   *
   * Calls `cudaStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(stream_)); }

  /**
   * @brief Synchronize the viewed CUDA stream. Does not throw if there is an error.
   *
   * Calls `cudaStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
  }

  /**
   * @brief Record the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaEventRecord()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cuda_event_view event) const
  {
    RMM_CUDA_TRY(cudaEventRecord(event.value(), value()));
  }

  /**
   * @brief Record the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cuda_event_view event, cuda_event_record_flags flags) const
  {
#if CUDART_VERSION < 11010
    RMM_CUDA_TRY(cudaEventRecord(event.value(), value()));
#else
    RMM_CUDA_TRY(cudaEventRecordWithFlags(event.value(), value(), flags));
#endif
  }

  /**
   * @brief Record the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaEventRecord()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cuda_event_view event) const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(event.value(), value()));
  }

  /**
   * @brief Record the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cuda_event_view event, cuda_event_record_flags flags) const noexcept
  {
#if CUDART_VERSION < 11010
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(event.value(), value()));
#else
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecordWithFlags(event.value(), value(), flags));
#endif
  }

  /**
   * @brief Wait for the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()`.
   *
   * @throw rmm::cuda_error if event waiting fails
   */
  void wait(cuda_event_view event, cuda_event_wait_flags flags = EVENT_WAIT_DEFAULT) const
  {
    RMM_CUDA_TRY(cudaStreamWaitEvent(value(), event.value(), flags));
  }

  /**
   * @brief Wait for the given CUDA event in the viewed CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()` asserting the CUDA_SUCCESS result.
   */
  void wait_no_throw(cuda_event_view event,
                     cuda_event_wait_flags flags = EVENT_WAIT_DEFAULT) const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(value(), event.value(), flags));
  }

 private:
  cudaStream_t stream_{0};
};

/**
 * @brief Static cuda_stream_view of the default stream (stream 0), for convenience
 */
static constexpr cuda_stream_view cuda_stream_default{};

/**
 * @brief Static cuda_stream_view of cudaStreamLegacy, for convenience
 */
static cuda_stream_view cuda_stream_legacy{cudaStreamLegacy};

/**
 * @brief Static cuda_stream_view of cudaStreamPerThread, for convenience
 */
static cuda_stream_view cuda_stream_per_thread{cudaStreamPerThread};

/**
 * @brief Equality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs)
{
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output stream operator for printing / logging streams
 *
 * @param os The output ostream
 * @param sv The cuda_stream_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_stream_view sv)
{
  os << sv.value();
  return os;
}

}  // namespace rmm
