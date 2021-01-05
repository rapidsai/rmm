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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <vector>

namespace rmm {

/**
 * @brief A pool of CUDA streams.
 *
 * Provides efficient access to collection of CUDA stream objects.
 *
 * Successive calls may return a `cuda_stream_view` of identical streams. For example, a possible
 * implementation is to maintain a circular buffer of `cuda_stream` objects.
 */
class cuda_stream_pool {
 public:
  static constexpr std::size_t default_size{16};  ///< Default stream pool size

  /**
   * @brief Construct a new cuda stream pool object of the given size
   *
   * @param pool_size The number of streams in the pool
   */
  explicit cuda_stream_pool(std::size_t pool_size = default_size) : streams_(pool_size) {}
  ~cuda_stream_pool() = default;

  cuda_stream_pool(cuda_stream_pool&&)      = delete;
  cuda_stream_pool(cuda_stream_pool const&) = delete;
  cuda_stream_pool& operator=(cuda_stream_pool&&) = delete;
  cuda_stream_pool& operator=(cuda_stream_pool const&) = delete;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return rmm::cuda_stream_view
   */
  rmm::cuda_stream_view get_stream() const noexcept
  {
    return streams_[(next_stream++) % streams_.size()].view();
  }

 private:
  std::vector<rmm::cuda_stream> streams_;
  mutable std::atomic_size_t next_stream{};
};

}  // namespace rmm
