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

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

/**
 * @brief Strongly-typed wrapper for CUDA streams with default constructor.
 */
class cuda_stream_view {
 public:
  cuda_stream_view(cuda_stream_view const&) = default;
  cuda_stream_view(cuda_stream_view&&)      = default;
  cuda_stream_view& operator=(cuda_stream_view const&) = default;
  cuda_stream_view& operator=(cuda_stream_view&&) = default;

  // TODO disable construction from 0 after cuDF and others adopt cuda_stream_view
  // cuda_stream_view(int)            = delete; //< Prevent cast from 0
  // cuda_stream_view(std::nullptr_t) = delete; //< Prevent cast from nullptr

  /**
   * @brief Construct a default cudaStream_t.
   */
  constexpr explicit cuda_stream_view() : _stream{0} {}

  /**
   * @brief Implicit conversion from cudaStream_t.
   */
  constexpr cuda_stream_view(cudaStream_t stream) : _stream{stream} {}

  /**
   * @brief Implicit conversion to cudaStream_t.
   */
  operator cudaStream_t() const { return _stream; }

  /**
   * @brief Explicit conversion to uintptr_t.
   */
  explicit operator uintptr_t() const { return reinterpret_cast<uintptr_t>(_stream); }

  /**
   * @brief Compare two streams for equality.
   */
  bool operator==(cuda_stream_view const& other) { return _stream == other._stream; }

 private:
  cudaStream_t _stream;
};

namespace detail {
// Use an atomic to guarantee thread safety
inline std::atomic<cuda_stream_view>& default_stream()
{
  static std::atomic<cuda_stream_view> res{cuda_stream_view{}};
  return res;
}
}  // namespace detail

/**
 * @brief Get the default stream view.
 *
 * The default stream view is used when an explicit stream view
 * is not supplied. The initial default stream view is cudaStreamDefault.
 *
 * This function is thread-safe.
 *
 * @return cuda_stream_view The current default stream view
 */
inline cuda_stream_view get_default_stream() { return detail::default_stream().load(); }

/**
 * @brief Sets the default stream view.
 *
 * This function is thread-safe.
 *
 * @param new_stream Stream view to use as new default stream view
 * @return The previous value of the default stream view
 */
inline cuda_stream_view set_default_stream(cuda_stream_view new_stream)
{
  return detail::default_stream().exchange(new_stream);
}

}  // namespace rmm
