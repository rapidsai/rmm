/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace rmm {

/**
 * @brief Strong type for a CUDA device identifier.
 *
 */
struct cuda_device_id {
  using value_type = int;

  /**
   * @brief Construct a `cuda_device_id` from the specified integer value
   *
   * @param id The device's integer identifier
   */
  explicit constexpr cuda_device_id(value_type dev_id) noexcept : id_{dev_id} {}

  /// Returns the wrapped integer value
  [[nodiscard]] constexpr value_type value() const noexcept { return id_; }

 private:
  value_type id_;
};

namespace detail {
/**
 * @brief Returns a `cuda_device_id` for the current device
 *
 * The current device is the device on which the calling thread executes device code.
 *
 * @return `cuda_device_id` for the current device
 */
inline cuda_device_id current_device()
{
  int dev_id{};
  RMM_CUDA_TRY(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}
}  // namespace detail
}  // namespace rmm
