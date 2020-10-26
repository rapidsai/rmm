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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <memory>

namespace rmm {

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
   */
  cuda_stream& operator=(cuda_stream&&) = default;
  ~cuda_stream()                        = default;
  cuda_stream(cuda_stream const&)       = delete;  // Copying disallowed: one stream one owner
  cuda_stream& operator=(cuda_stream&) = delete;

  /**
   * @brief Construct a new cuda stream object
   *
   * @throw rmm::cuda_error if stream creation fails
   */
  cuda_stream()
    : stream_{[]() {
                cudaStream_t* s = new cudaStream_t;
                RMM_CUDA_TRY(cudaStreamCreate(s));
                return s;
              }(),
              [](cudaStream_t* s) {
                RMM_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(*s));
                delete s;
              }}
  {
  }

  /**
   * @brief Returns true if the owned stream is non-null
   *
   * @return true If the owned stream has not been explicitly moved and is therefore non-null.
   * @return false If the owned stream has been explicitly moved and is therefore null.
   */
  bool is_valid() const { return stream_ != nullptr; }

  /**
   * @brief Get the value of the wrapped CUDA stream.
   *
   * @return cudaStream_t The wrapped CUDA stream.
   */
  cudaStream_t value() const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return *stream_;
  }

  /**
   * @brief Explicit conversion to cudaStream_t.
   */
  explicit operator cudaStream_t() const noexcept { return value(); }

  /**
   * @brief Creates an immutable, non-owning view of the wrapped CUDA stream.
   *
   * @return rmm::cuda_stream_view The view of the CUDA stream
   */
  cuda_stream_view view() const { return cuda_stream_view{value()}; }

  /**
   * @brief Implicit conversion to cuda_stream_view
   *
   * @return A view of the owned stream
   */
  operator cuda_stream_view() const { return view(); }

  /**
   * @brief Synchronize the owned CUDA stream.
   *
   * Calls `cudaStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(value())); }

  /**
   * @brief Synchronize the owned CUDA stream. Does not throw if there is an error.
   *
   * Calls `cudaStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(value()));
  }

 private:
  std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> stream_;
};

}  // namespace rmm
