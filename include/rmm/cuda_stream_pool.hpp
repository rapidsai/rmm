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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

namespace rmm {
/**
 * @addtogroup cuda_streams
 * @{
 * @file
 */

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
   * @brief Construct a new cuda stream pool object of the given non-zero size
   *
   * @throws logic_error if `pool_size` is zero
   * @param pool_size The number of streams in the pool
   */
  explicit cuda_stream_pool(std::size_t pool_size = default_size) : streams_(pool_size)
  {
    RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  }
  ~cuda_stream_pool() = default;

  cuda_stream_pool(cuda_stream_pool&&)                 = delete;
  cuda_stream_pool(cuda_stream_pool const&)            = delete;
  cuda_stream_pool& operator=(cuda_stream_pool&&)      = delete;
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

  /**
   * @brief Get a `cuda_stream_view` of the stream associated with `stream_id`.
   * Equivalent values of `stream_id` return a stream_view to the same underlying stream.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param stream_id Unique identifier for the desired stream
   *
   * @return rmm::cuda_stream_view
   */
  rmm::cuda_stream_view get_stream(std::size_t stream_id) const
  {
    return streams_[stream_id % streams_.size()].view();
  }

  /**
   * @brief Get the number of streams in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of streams in the pool
   */
  std::size_t get_pool_size() const noexcept { return streams_.size(); }

 private:
  std::vector<rmm::cuda_stream> streams_;
  mutable std::atomic_size_t next_stream{};
};

/** @} */  // end of group
}  // namespace rmm
