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

#include <cstddef>

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

/**
 * @brief Strongly-typed wrapper for CUDA streams with default constructor.
 */
class stream_t {
 public:
  stream_t(stream_t const&) = default;
  stream_t(stream_t&&)      = default;
  stream_t& operator=(stream_t const&) = default;
  stream_t& operator=(stream_t&&) = default;

  // TODO disable construction from 0 after cuDF and others adopt stream_t
  //stream_t(int)            = delete; //< Prevent cast from 0
  //stream_t(std::nullptr_t) = delete; //< Prevent cast from nullptr

  /**
   * @brief Construct a default cudaStream_t.
   */
  constexpr explicit stream_t(): _stream{0} {}

  /**
   * @brief Implicitly convert from cudaStream_t.
   */
  stream_t(cudaStream_t stream) : _stream{stream} {}

  /**
   * @brief Implicitly convert to cudaStream_t.
   */
  operator cudaStream_t() const { return _stream; }

  /**
   * @brief Compare two streams for equality.
   */
  bool operator==(stream_t const& other) { return _stream == other._stream; }

 private:
  cudaStream_t _stream;
};

}  // namespace rmm
