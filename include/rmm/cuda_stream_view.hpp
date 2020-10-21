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

  /**
   * @brief Implicit conversion from cudaStream_t.
   */
  constexpr cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

  /**
   * @brief Implicit conversion to cudaStream_t.
   */
  constexpr operator cudaStream_t() const noexcept { return stream_; }

  /**
   * @brief Explicit conversion to uintptr_t.
   */
  explicit operator uintptr_t() const noexcept { return reinterpret_cast<uintptr_t>(stream_); }

  /**
   * @brief Compare two streams for equality.
   */
  constexpr bool operator==(cuda_stream_view const& other) const noexcept
  {
    return stream_ == other.stream_;
  }

  /**
   * @brief Synchronize the viewed CUDA stream.
   *
   * Calls `cudaStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() { RMM_CUDA_TRY(cudaStreamSynchronize(stream_)); }

 private:
  cudaStream_t stream_{cudaStreamDefault};
};

}  // namespace rmm
