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

#include <mutex>

namespace rmm {
namespace mr {

/**
 * @brief Strong type for a CUDA device identifier.
 *
 */
enum class cuda_device_id : int {};

namespace detail {
std::mutex& map_lock()
{
  static std::mutex map_lock;
  return map_lock;
}

auto& get_map()
{
  static std::unordered_map<cuda_device_id, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

/**
 * @brief Returns a `cuda_device_id` for the current device
 *
 * The current device is the device on which the calling thread executes device code.
 *
 * @return `cuda_device_id` for the current device
 */
cuda_device_id current_device()
{
  int dev_id;
  RMM_CUDA_TRY(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}
}  // namespace detail

/**
 * @brief Get the resource for the specified device
 *
 * Returns a pointer to the resource set for the specified device. The initial resource is a
 * `cuda_memory_resource`.
 *
 * This function is thread-safe.
 *
 * @param id The id of the target device
 * @return Pointer to the current resource for `id`
 */
inline device_memory_resource* get_per_device_resource(cuda_device_id id)
{
  std::lock_guard<std::mutex> lock{map_lock()};
  auto& map        = get_map();
  auto const found = map.find(id);
  // If a resource was never set for `id`, set to the initial resource
  return (found == map.end()) ? (map[id] = initial_resource()) : map[id];
}

/**
 * @brief Update the resource for the specified device
 *
 * If `new_mr` is not `nullptr`, sets the resource pointer for the device specified by `id` to
 * `new_mr`. Otherwise, resets `id`s resource to the initial `cuda_memory_resource`.
 *
 * The object pointed to by `new_mr` must outlive the last use of the resource, else behavior is
 * undefined. It is the caller's responsibility to maintain the lifetime of the resource object.
 *
 * This function is thread-safe.
 *
 * @param new_mr If not `nullptr`, pointer to new `device_memory_resource` to use as new resource
 * for `id`
 * @return Pointer to the previous the resource for `id`
 */
inline device_memory_resource* set_per_device_resource(cuda_device_id,
                                                       devicce_memory_resource* new_mr)
{
  std::lock_guard<std::mutex> lock{map_lock()};
  auto& map          = get_map();
  auto const old_itr = map.find(id);
  // If a resource didn't previously exist for `id`, return pointer to initial_resource
  auto old_mr = (old_itr == map.end()) ? detail::initial_resource() : old_itr->second;
  map[id]     = (new_mr == nullptr) ? detail::initial_resource() : new_mr;
  return old_mr;
}

/**
 * @brief Get the resource for the current device
 *
 * Returns a pointer to the resource set for the current device. The initial resource is a
 * `cuda_memory_resource`.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * This function is thread-safe
 *
 * @return Pointer to the resource for the current device
 */
inline device_memory_resource* get_current_device_resource()
{
  return get_per_device_resource(current_device());
}

/**
 * @brief Update the resource for the current device
 *
 * If `new_mr` is not `nullptr`, sets the resource pointer for the current device to
 * `new_mr`. Otherwise, resets the resource to the initial `cuda_memory_resource`.

 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * The object pointed to by `new_mr` must outlive the last use of the resource, else behavior is
 * undefined. It is the caller's responsibility to maintain the lifetime of the resource object.
 *
 * This function is thread-safe.
 *
 * @param new_mr If not `nullptr`, pointer to new resource to use for the current device
 * @return Pointer to the previous resource for the current device
 */
inline device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)
{
  return set_per_device_resource(current_device(), new_mr);
}
}  // namespace mr
}  // namespace rmm
