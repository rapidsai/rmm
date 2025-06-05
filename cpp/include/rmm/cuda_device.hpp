/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>

#include <cstddef>
#include <utility>

namespace RMM_EXPORT rmm {

struct cuda_device_id;
cuda_device_id get_current_cuda_device();

/**
 * @addtogroup cuda_device_management
 * @{
 * @file
 */
/**
 * @brief Strong type for a CUDA device identifier.
 *
 */
struct cuda_device_id {
  using value_type = int;  ///< Integer type used for device identifier

  /**
   * @brief Construct a `cuda_device_id` from the current device
   */
  cuda_device_id() noexcept : id_{get_current_cuda_device().value()} {}

  /**
   * @brief Construct a `cuda_device_id` from the specified integer value.
   *
   * @param dev_id The device's integer identifier
   */
  explicit constexpr cuda_device_id(value_type dev_id) noexcept : id_{dev_id} {}

  /// @briefreturn{The wrapped integer value}
  [[nodiscard]] constexpr value_type value() const noexcept { return id_; }

  // TODO re-add doxygen comment specifier /** for these hidden friend operators once this Breathe
  // bug is fixed: https://github.com/breathe-doc/breathe/issues/916
  //! @cond Doxygen_Suppress
  /**
   * @brief Compare two `cuda_device_id`s for equality.
   *
   * @param lhs The first `cuda_device_id` to compare.
   * @param rhs The second `cuda_device_id` to compare.
   * @return true if the two `cuda_device_id`s wrap the same integer value, false otherwise.
   */
  [[nodiscard]] constexpr friend bool operator==(cuda_device_id const& lhs,
                                                 cuda_device_id const& rhs) noexcept
  {
    return lhs.value() == rhs.value();
  }

  /**
   * @brief Compare two `cuda_device_id`s for inequality.
   *
   * @param lhs The first `cuda_device_id` to compare.
   * @param rhs The second `cuda_device_id` to compare.
   * @return true if the two `cuda_device_id`s wrap different integer values, false otherwise.
   */
  [[nodiscard]] constexpr friend bool operator!=(cuda_device_id const& lhs,
                                                 cuda_device_id const& rhs) noexcept
  {
    return lhs.value() != rhs.value();
  }
  //! @endcond
 private:
  value_type id_;
};

/**
 * @brief Returns a `cuda_device_id` for the current device
 *
 * The current device is the device on which the calling thread executes device code.
 *
 * @return `cuda_device_id` for the current device
 */
cuda_device_id get_current_cuda_device();

/**
 * @brief Returns the number of CUDA devices in the system
 *
 * @return Number of CUDA devices in the system
 */
int get_num_cuda_devices();

/**
 * @brief Returns the available and total device memory in bytes for the current device
 *
 * @return The available and total device memory in bytes for the current device as a std::pair.
 */
std::pair<std::size_t, std::size_t> available_device_memory();

/**
 * @brief Returns the approximate specified percent of available device memory on the current CUDA
 * device, aligned (down) to the nearest CUDA allocation size.
 *
 * @param percent The percent of free memory to return.
 *
 * @return The recommended initial device memory pool size in bytes.
 */
std::size_t percent_of_free_device_memory(int percent);

/**
 * @brief RAII class that sets the current CUDA device to the specified device on construction
 * and restores the previous device on destruction.
 */
struct cuda_set_device_raii {
  /**
   * @brief Construct a new cuda_set_device_raii object and sets the current CUDA device to `dev_id`
   *
   * @param dev_id The device to set as the current CUDA device
   */
  explicit cuda_set_device_raii(cuda_device_id dev_id);
  /**
   * @brief Reactivates the previous CUDA device
   */
  ~cuda_set_device_raii() noexcept;

  cuda_set_device_raii(cuda_set_device_raii const&)            = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii const&) = delete;
  cuda_set_device_raii(cuda_set_device_raii&&)                 = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii&&)      = delete;

 private:
  cuda_device_id old_device_;
  bool needs_reset_;
};

/** @} */  // end of group
}  // namespace RMM_EXPORT rmm
