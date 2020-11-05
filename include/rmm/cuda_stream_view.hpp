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

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

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

  // TODO disable construction from 0 after cuDF and others adopt cuda_stream_view
  // cuda_stream_view(int)            = delete; //< Prevent cast from 0
  // cuda_stream_view(std::nullptr_t) = delete; //< Prevent cast from nullptr
  // TODO also disable implicit conversion from cudaStream_t

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
   * @brief Explicit conversion to cudaStream_t.
   */
  explicit constexpr operator cudaStream_t() const noexcept { return value(); }

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
