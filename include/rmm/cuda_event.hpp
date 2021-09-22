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
#include <rmm/cuda_event_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <memory>

namespace rmm {

/** @brief An event view with flags provided at runtime. */
class cuda_event_ {
 public:
  /**
   * @brief Move constructor (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event_(cuda_event_&&) = default;
  /**
   * @brief Move copy assignment operator (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event_& operator=(cuda_event_&&) = default;
  ~cuda_event_()                        = default;
  cuda_event_(cuda_event_ const&)       = delete;  // Copying disallowed: one event one owner
  cuda_event_& operator=(cuda_event_&) = delete;

  /**
   * @brief Construct a new cuda event object specifying flags at runtime.
   *
   * @throw rmm::cuda_error if event creation fails
   */
  cuda_event_(cuda_event_flags flags)
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
   * @brief Creates an immutable, non-owning view of the wrapped CUDA event.
   *
   * @return rmm::cuda_event_view The view of the CUDA event
   */
  cuda_event_view_ view() const { return cuda_event_view_(value()); }

  /**
   * @brief Implicit conversion to cuda_event_view
   *
   * @return A view of the owned event
   */
  operator cuda_event_view_() const { return view(); }

  /**
   * @brief Computes the elapsed time between since the given event till the owned event.
   *
   * @return time in milliseconds.
   */
  float elapsed_time_since(const cuda_event_view_& prev_event) const
  {
    return view().elapsed_time_since(prev_event);
  }

  /**
   * @brief Queries the owned event's status.
   *
   * @return whether all captured work has been completed.
   */
  bool query_status() const { return view().query_status(); }

  /**
   * @brief Synchronize the owned CUDA event (block host thread till completion).
   *
   * Calls `cudaEventSynchronize()`.
   *
   * @throw rmm::cuda_error if event synchronization fails
   */
  void wait() const { view().wait(); }

  /**
   * @brief Synchronize the owned CUDA event (block host thread till completion).
   *
   * Calls `cudaEventSynchronize()` asserting the CUDA_SUCCESS result
   */
  void wait_no_throw() const noexcept { view().wait_no_throw(); }

 private:
  std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>> event_;
};

/**
 * @brief Owning wrapper for a CUDA event.
 *
 * Provides RAII lifetime semantics for a CUDA event.
 */
template <cuda_event_flags Flags = EVENT_DEFAULT>
class cuda_event : public cuda_event_ {
 public:
  /**
   * @brief Move constructor (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event(cuda_event<Flags>&&) = default;
  /**
   * @brief Move copy assignment operator (default)
   *
   * A moved-from cuda_event is invalid and it is Undefined Behavior to call methods that access
   * the owned event.
   */
  cuda_event& operator=(cuda_event<Flags>&&) = default;
  ~cuda_event()                              = default;
  cuda_event(cuda_event<Flags> const&)       = delete;  // Copying disallowed: one event one owner
  cuda_event& operator=(cuda_event<Flags>&) = delete;

  /**
   * @brief Construct a new cuda event object
   *
   * @throw rmm::cuda_error if event creation fails
   */
  cuda_event() : cuda_event_(Flags) {}

  /**
   * @brief Creates an immutable, non-owning view of the wrapped CUDA event.
   *
   * @return rmm::cuda_event_view The view of the CUDA event
   */
  cuda_event_view<Flags> view() const { return cuda_event_view<Flags>(*this); }

  /**
   * @brief Implicit conversion to cuda_event_view
   *
   * @return A view of the owned event
   */
  operator cuda_event_view<Flags>() const { return view(); }

  /**
   * @brief Computes the elapsed time between since the given event till the owned event.
   *
   * @return time in milliseconds.
   */
  float elapsed_time_since(const cuda_event_view_& prev_event) const
  {
    return view().elapsed_time_since(prev_event);
  }

  /**
   * @brief Computes the elapsed time between since the given event till the owned event.
   *
   * @return time in milliseconds.
   */
  template <cuda_event_flags FlagsPrev>
  float elapsed_time_since(const cuda_event_view<FlagsPrev>& prev_event) const
  {
    return view().elapsed_time_since(prev_event);
  }
};

template <cuda_event_flags Flags>
constexpr cuda_event_view<Flags>::cuda_event_view(const cuda_event<Flags>& event) noexcept
  : cuda_event_view_(event.value())
{
}

}  // namespace rmm
