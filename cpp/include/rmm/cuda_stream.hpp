/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>

#include <cuda_runtime_api.h>

#include <functional>
#include <memory>

namespace RMM_EXPORT rmm {
/**
 * @addtogroup cuda_streams
 * @{
 * @file
 */

/**
 * @brief Owning wrapper for a CUDA stream.
 *
 * Provides RAII lifetime semantics for a CUDA stream.
 */
class cuda_stream {
 public:
  /**
   * @brief Move constructor (default)
   *
   * A moved-from cuda_stream is invalid and it is Undefined Behavior to call methods that access
   * the owned stream.
   */
  cuda_stream(cuda_stream&&) = default;
  /**
   * @brief Move copy assignment operator (default)
   *
   * A moved-from cuda_stream is invalid and it is Undefined Behavior to call methods that access
   * the owned stream.
   *
   * @return A reference to this cuda_stream
   */
  cuda_stream& operator=(cuda_stream&&) = default;
  ~cuda_stream()                        = default;
  cuda_stream(cuda_stream const&)       = delete;  // Copying disallowed: one stream one owner
  cuda_stream& operator=(cuda_stream&)  = delete;

  /**
   * @brief Construct a new cuda stream object
   *
   * @throw rmm::cuda_error if stream creation fails
   */
  cuda_stream();

  /**
   * @brief Returns true if the owned stream is non-null
   *
   * @return true If the owned stream has not been explicitly moved and is therefore non-null.
   * @return false If the owned stream has been explicitly moved and is therefore null.
   */
  [[nodiscard]] bool is_valid() const;

  /**
   * @brief Get the value of the wrapped CUDA stream.
   *
   * @return cudaStream_t The wrapped CUDA stream.
   */
  [[nodiscard]] cudaStream_t value() const;

  /**
   * @brief Explicit conversion to cudaStream_t.
   */
  explicit operator cudaStream_t() const noexcept;

  /**
   * @brief Creates an immutable, non-owning view of the wrapped CUDA stream.
   *
   * @return rmm::cuda_stream_view The view of the CUDA stream
   */
  [[nodiscard]] cuda_stream_view view() const;

  /**
   * @brief Implicit conversion to cuda_stream_view
   *
   * @return A view of the owned stream
   */
  operator cuda_stream_view() const;

  /**
   * @brief Synchronize the owned CUDA stream.
   *
   * Calls `cudaStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const;

  /**
   * @brief Synchronize the owned CUDA stream. Does not throw if there is an error.
   *
   * Calls `cudaStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept;

 private:
  std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> stream_;
};

/** @} */  // end of group
}  // namespace RMM_EXPORT rmm
