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

#include "default_memory_resource.hpp"
#include "device_memory_resource.hpp"

#include <map>
#include <mutex>

namespace rmm {
namespace mr {

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
  explicit constexpr cuda_device_id(value_type id) noexcept : id_{id} {}

  /// Returns the wrapped integer value
  constexpr value_type value() const noexcept { return id_; }

 private:
  value_type id_;
};

namespace detail {
inline std::mutex& map_lock()
{
  static std::mutex map_lock;
  return map_lock;
}

inline auto& get_map()
{
  static std::map<cuda_device_id::value_type, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

/**
 * @brief Returns a `cuda_device_id` for the current device
 *
 * The current device is the device on which the calling thread executes device code.
 *
 * @return `cuda_device_id` for the current device
 */
inline cuda_device_id current_device()
{
  int dev_id;
  RMM_CUDA_TRY(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}
}  // namespace detail

/**
 * @brief Get the resource for the specified device.
 *
 * Returns a pointer to the `device_memory_resource` for the specified device. The initial resource
 * is a `cuda_memory_resource`.
 *
 * This function is thread-safe with respect to `set_per_device_resource`,
 * `get_current_device_resource`, and `set_current_device_resource`. Concurrent calls to any of
 * these functions will result in a valid state, but the order of execution is undefined.
 *
 * @param id The id of the target device
 * @return Pointer to the current `device_memory_resource` for device `id`
 */
inline device_memory_resource* get_per_device_resource(cuda_device_id id)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};
  auto& map = detail::get_map();
  // If a resource was never set for `id`, set to the initial resource
  auto found = map.find(id.value());
  return (found == map.end()) ? (map[id.value()] = detail::initial_resource()) : found->second;
}

/**
 * @brief Set the `device_memory_resource` for the specified device.
 *
 * If `new_mr` is not `nullptr`, sets the memory resource pointer for the device specified by `id`
 * to `new_mr`. Otherwise, resets `id`s resource to the initial `cuda_memory_resource`.
 *
 * The object pointed to by `new_mr` must outlive the last use of the resource, otherwise behavior
 * is undefined. It is the caller's responsibility to maintain the lifetime of the resource object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @param new_mr If not `nullptr`, pointer to new `device_memory_resource` to use as new resource
 * for `id`
 * @return Pointer to the previous memory resource for `id`
 */
inline device_memory_resource* set_per_device_resource(cuda_device_id id,
                                                       device_memory_resource* new_mr)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};
  auto& map          = detail::get_map();
  auto const old_itr = map.find(id.value());
  // If a resource didn't previously exist for `id`, return pointer to initial_resource
  auto old_mr     = (old_itr == map.end()) ? detail::initial_resource() : old_itr->second;
  map[id.value()] = (new_mr == nullptr) ? detail::initial_resource() : new_mr;
  return old_mr;
}

/**
 * @brief Get the memory resource for the current device.
 *
 * Returns a pointer to the resource set for the current device. The initial resource is a
 * `cuda_memory_resource`.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @return Pointer to the resource for the current device
 */
inline device_memory_resource* get_current_device_resource()
{
  return get_per_device_resource(detail::current_device());
}

/**
 * @brief Set the memory resource for the current device.
 *
 * If `new_mr` is not `nullptr`, sets the resource pointer for the current device to
 * `new_mr`. Otherwise, resets the resource to the initial `cuda_memory_resource`.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * The object pointed to by `new_mr` must outlive the last use of the resource, otherwise behavior
 * is  undefined. It is the caller's responsibility to maintain the lifetime of the resource
 * object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @param new_mr If not `nullptr`, pointer to new resource to use for the current device
 * @return Pointer to the previous resource for the current device
 */
inline device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)
{
  return set_per_device_resource(detail::current_device(), new_mr);
}
}  // namespace mr
}  // namespace rmm
