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
 *
 */
class cuda_stream {
 public:
  // No copy construction or assignment
  cuda_stream(cuda_stream const&) = delete;
  cuda_stream& operator=(cuda_stream const&) = delete;

  // Move construction and assignment allowed
  cuda_stream(cuda_stream&&) = default;
  cuda_stream& operator=(cuda_stream&&) = default;

  ~cuda_stream() = default;

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
              [](cudaStream_t* s) { RMM_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(*s)); }}
  {
  }

  /**
   * @brief Creates an immutable, non-owning view of the CUDA stream.
   *
   * @return rmm::cuda_stream_view The view of the CUDA stream
   */
  rmm::cuda_stream_view view() const { return rmm::cuda_stream_view{*stream_}; }

  /**
   * @brief Implicit conversion to cudaStream_t.
   */
  operator cudaStream_t() const noexcept { return *stream_; }

  /**
   * @brief Implicit conversion to cuda_stream_view
   *
   * @return A view of the owned stream
   */
  operator cuda_stream_view() const { return view(); }

  /**
   * @brief Synchronize the owned CUDA stream.
   *
   * Call's `cudaStreamSynchronize()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() { RMM_CUDA_TRY(cudaStreamSynchronize(*stream_)); }

  /**
   * @brief Compare two streams for equality.
   */
  bool operator==(cuda_stream const& other) const noexcept { return stream_ == other.stream_; }

  /**
   * @brief Compare two streams for equality.
   */
  bool operator==(cuda_stream_view const& other) const noexcept { return *stream_ == other; }

 private:
  std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> stream_;
};

}  // namespace rmm
