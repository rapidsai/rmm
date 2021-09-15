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

#include <cuda_runtime_api.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <memory>

namespace rmm {

/**
 * @brief Owning wrapper for a CUDA event.
 *
 * Provides RAII lifetime semantics for a CUDA event.
 */
class cuda_event {
 public:
  /**
   * @brief Move constructor (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event(cuda_event&&) = default;
  /**
   * @brief Move copy assignment operator (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event& operator=(cuda_event&&) = default;
  ~cuda_event()                       = default;
  cuda_event(cuda_event const&)       = delete;  // Copying disallowed: one event one owner
  cuda_event& operator=(cuda_event&) = delete;

  /**
   * @brief Construct a new cuda event object
   *
   * @throw rmm::cuda_error if event creation fails
   */
  cuda_event()
    : event_{[]() {
               cudaEvent_t* s = new cudaEvent_t;
               RMM_CUDA_TRY(cudaEventCreate(s));
               return s;
             }(),
             [](cudaEvent_t* s) {
               RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(*s));
               delete s;
             }}
  {
  }

  /**
   * @brief Construct a new cuda event object
   *
   * @throw rmm::cuda_error if event creation fails
   */
  cuda_event(unsigned int flags)
    : event_{[flags]() {
               cudaEvent_t* s = new cudaEvent_t;
               RMM_CUDA_TRY(cudaEventCreateWithFlags(s, flags));
               return s;
             }(),
             [](cudaEvent_t* s) {
               RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(*s));
               delete s;
             }}
  {
  }

  /**
   * @brief Returns true if the owned event is non-null
   *
   * @return true If the owned event has not been explicitly moved and is therefore non-null.
   * @return false If the owned event has been explicitly moved and is therefore null.
   */
  bool is_valid() const { return event_ != nullptr; }

  /**
   * @brief Get the value of the wrapped CUDA event.
   *
   * @return cudaEvent_t The wrapped CUDA event.
   */
  cudaEvent_t value() const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return *event_;
  }

  /**
   * @brief Explicit conversion to cudaEvent_t.
   */
  explicit operator cudaEvent_t() const noexcept { return value(); }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecord()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cudaStream_t stream) const { RMM_CUDA_TRY(cudaEventRecord(value(), stream)); }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecord()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cuda_stream_view stream) const { record(stream.value()); }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cudaStream_t stream, unsigned int flags) const
  {
#if CUDART_VERSION < 11010
    RMM_CUDA_TRY(cudaEventRecord(value(), stream));
#else
    RMM_CUDA_TRY(cudaEventRecordWithFlags(value(), stream, flags));
#endif
  }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void record(cuda_stream_view stream, unsigned int flags) const { record(stream.value(), flags); }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecord()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cudaStream_t stream) const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(value(), stream));
  }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecord()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cuda_stream_view stream) const noexcept { record_no_throw(stream.value()); }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cudaStream_t stream, unsigned int flags) const noexcept
  {
#if CUDART_VERSION < 11010
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(value(), stream));
#else
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecordWithFlags(value(), stream, flags));
#endif
  }

  /**
   * @brief Record the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaEventRecordWithFlags()` asserting the CUDA_SUCCESS result.
   */
  void record_no_throw(cuda_stream_view stream, unsigned int flags) const noexcept
  {
    record_no_throw(stream.value(), flags);
  }

  /**
   * @brief Wait for the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()`.
   *
   * @throw rmm::cuda_error if event waiting fails
   */
  void wait(cudaStream_t stream, unsigned int flags = 0u) const
  {
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream, value(), flags));
  }

  /**
   * @brief Wait for the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()`.
   *
   * @throw rmm::cuda_error if event waiting fails
   */
  void wait(cuda_stream_view stream, unsigned int flags = 0u) const { wait(stream.value(), flags); }

  /**
   * @brief Wait for the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()` asserting the CUDA_SUCCESS result.
   */
  void wait_no_throw(cudaStream_t stream, unsigned int flags = 0u) const noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(stream, value(), flags));
  }

  /**
   * @brief Wait for the owned CUDA event in the given CUDA stream.
   *
   * Calls `cudaStreamWaitEvent()` asserting the CUDA_SUCCESS result.
   */
  void wait_no_throw(cuda_stream_view stream, unsigned int flags = 0u) const noexcept
  {
    wait_no_throw(stream.value(), flags);
  }

  /**
   * @brief Wait for the owned CUDA event on the host.
   *
   * Calls `cudaEventSynchronize()`.
   *
   * @throw rmm::cuda_error if event recording fails
   */
  void wait() const { RMM_CUDA_TRY(cudaEventSynchronize(value())); }

  /**
   * @brief Wait for the owned CUDA event on the host.
   *
   * Calls `cudaEventSynchronize()` asserting the CUDA_SUCCESS result.
   */
  void wait_no_throw() const noexcept { RMM_ASSERT_CUDA_SUCCESS(cudaEventSynchronize(value())); }

 private:
  std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>> event_;
};

}  // namespace rmm
