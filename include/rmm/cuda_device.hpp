/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

namespace rmm {

struct cuda_device_id;
inline cuda_device_id get_current_cuda_device();

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
inline cuda_device_id get_current_cuda_device()
{
  cuda_device_id::value_type dev_id{-1};
  RMM_ASSERT_CUDA_SUCCESS(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}

/**
 * @brief Returns the number of CUDA devices in the system
 *
 * @return Number of CUDA devices in the system
 */
inline int get_num_cuda_devices()
{
  cuda_device_id::value_type num_dev{-1};
  RMM_ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&num_dev));
  return num_dev;
}

/**
 * @brief Returns the available and total device memory in bytes for the current device
 *
 * @return The available and total device memory in bytes for the current device as a std::pair.
 */
inline std::pair<std::size_t, std::size_t> available_device_memory()
{
  std::size_t free{};
  std::size_t total{};
  RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
  return {free, total};
}

/**
 * @brief Returns the approximate specified percent of available device memory on the current CUDA
 * device, aligned (down) to the nearest CUDA allocation size.
 *
 * @param percent The percent of free memory to return.
 *
 * @return The recommended initial device memory pool size in bytes.
 */
inline std::size_t percent_of_free_device_memory(int percent)
{
  [[maybe_unused]] auto const [free, total] = rmm::available_device_memory();
  auto fraction                             = static_cast<double>(percent) / 100.0;
  return rmm::align_down(static_cast<std::size_t>(static_cast<double>(free) * fraction),
                         rmm::CUDA_ALLOCATION_ALIGNMENT);
}

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
  explicit cuda_set_device_raii(cuda_device_id dev_id)
    : old_device_{get_current_cuda_device()},
      needs_reset_{dev_id.value() >= 0 && old_device_ != dev_id}
  {
    if (needs_reset_) { RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(dev_id.value())); }
  }
  /**
   * @brief Reactivates the previous CUDA device
   */
  ~cuda_set_device_raii() noexcept
  {
    if (needs_reset_) { RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(old_device_.value())); }
  }

  cuda_set_device_raii(cuda_set_device_raii const&)            = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii const&) = delete;
  cuda_set_device_raii(cuda_set_device_raii&&)                 = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii&&)      = delete;

 private:
  cuda_device_id old_device_;
  bool needs_reset_;
};

/** @} */  // end of group
}  // namespace rmm
