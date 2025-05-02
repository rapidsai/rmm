/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <map>
#include <mutex>

/**
 * @file per_device_resource.hpp
 * @brief Management of per-device `device_memory_resource`s
 *
 * One might wish to construct a `device_memory_resource` and use it for (de)allocation
 * without explicit dependency injection, i.e., passing a reference to that object to all places it
 * is to be used. Instead, one might want to set their resource as a "default" and have it be used
 * in all places where another resource has not been explicitly specified. In applications with
 * multiple GPUs in the same process, it may also be necessary to maintain independent default
 * resources for each device. To this end, the `set_per_device_resource` and
 * `get_per_device_resource` functions enable mapping a CUDA device id to a `device_memory_resource`
 * pointer.
 *
 * For example, given a pointer, `mr`, to a `device_memory_resource` object, calling
 * `set_per_device_resource(cuda_device_id{0}, mr)` will establish a mapping between CUDA device 0
 * and `mr` such that all future calls to `get_per_device_resource(cuda_device_id{0})` will return
 * the same pointer, `mr`. In this way, all places that use the resource returned from
 * `get_per_device_resource` for (de)allocation will use the user provided resource, `mr`.
 *
 * @note `device_memory_resource`s make CUDA API calls without setting the current CUDA device.
 * Therefore a memory resource should only be used with the current CUDA device set to the device
 * that was active when the memory resource was created. Calling `set_per_device_resource(id, mr)`
 * is only valid if `id` refers to the CUDA device that was active when `mr` was created.
 *
 * If no resource was explicitly set for a given device specified by `id`, then
 * `get_per_device_resource(id)` will return a pointer to a `cuda_memory_resource`.
 *
 * To fetch and modify the resource for the current CUDA device, `get_current_device_resource()` and
 * `set_current_device_resource()` automatically use the current CUDA device id from
 * `cudaGetDevice()`.
 *
 * RMM is in transition to use `cuda::mr::async_resource_ref` in place of raw pointers to
 * `device_memory_resource`. The `set_per_device_resource_ref`, `get_per_device_resource_ref`,
 * `get_current_device_resource_ref`, `set_current_device_resource_ref`, and
 * `reset_current_device_resource_ref` functions provide the same functionality as their
 * `device_memory_resource` counterparts, but with `device_async_resource_ref` objects. The raw
 * pointer versions and the `resource_ref` versions maintain distinc state and are not
 * interchangeable. The raw pointer versions are expected to be deprecated and removed in a future
 * release.
 *
 * Creating a device_memory_resource for each device requires care to set the current device
 * before creating each resource, and to maintain the lifetime of the resources as long as they
 * are set as per-device resources. Here is an example loop that creates `unique_ptr`s to
 * pool_memory_resource objects for each device and sets them as the per-device resource for that
 * device.
 *
 * @code{.cpp}
 * std::vector<unique_ptr<pool_memory_resource>> per_device_pools;
 * for(int i = 0; i < N; ++i) {
 *   cudaSetDevice(i);
 *   per_device_pools.push_back(std::make_unique<pool_memory_resource>());
 *   set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
 * }
 * @endcode
 * @code{.cpp}
 * using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
 * std::vector<unique_ptr<pool_mr>> per_device_pools;
 * for(int i = 0; i < N; ++i) {
 *   cudaSetDevice(i);
 *   // Note: for brevity, omitting creation of upstream and computing initial_size
 *   per_device_pools.push_back(std::make_unique<pool_mr>(upstream, initial_size));
 *   set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
 * }
 * @endcode
 */

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 */

namespace detail {

// These symbols must have default visibility so that when they are
// referenced in multiple different DSOs the linker correctly
// determines that there is only a single unique reference to the
// function symbols (and hence they return unique static references
// across different DSOs). See also
// https://github.com/rapidsai/rmm/issues/826
// Although currently the entire RMM namespace is RMM_EXPORT, we
// explicitly mark these functions as exported in case the namespace
// export changes.
/**
 * @brief Returns a pointer to the initial resource.
 *
 * Returns a global instance of a `cuda_memory_resource` as a function local static.
 *
 * @return Pointer to the static cuda_memory_resource used as the initial, default resource
 */
RMM_EXPORT inline device_memory_resource* initial_resource()
{
  static cuda_memory_resource mr{};
  return &mr;
}

/**
 * @briefreturn{Reference to the lock}
 */
RMM_EXPORT inline std::mutex& map_lock()
{
  static std::mutex map_lock;
  return map_lock;
}

/**
 * @briefreturn{Reference to the map from device id -> resource}
 */
RMM_EXPORT inline auto& get_map()
{
  static std::map<cuda_device_id::value_type, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

/**
 * @briefreturn{Reference to the lock}
 */
RMM_EXPORT inline std::mutex& ref_map_lock()
{
  static std::mutex ref_map_lock;
  return ref_map_lock;
}

// This symbol must have default visibility, see: https://github.com/rapidsai/rmm/issues/826
/**
 * @briefreturn{Reference to the map from device id -> resource_ref}
 */
RMM_EXPORT inline auto& get_ref_map()
{
  static std::map<cuda_device_id::value_type, device_async_resource_ref> device_id_to_resource_ref;
  return device_id_to_resource_ref;
}

}  // namespace detail

/**
 * @brief Get the resource for the specified device.
 *
 * Returns a pointer to the `device_memory_resource` for the specified device. The initial
 * resource is a `cuda_memory_resource`.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @note The returned `device_memory_resource` should only be used when CUDA device `device_id` is
 * the current device  (e.g. set using `cudaSetDevice()`). The behavior of a
 * `device_memory_resource` is undefined if used while the active CUDA device is a different device
 * from the one that was active when the `device_memory_resource` was created.
 *
 * @param device_id The id of the target device
 * @return Pointer to the current `device_memory_resource` for device `device_id`
 */
inline device_memory_resource* get_per_device_resource(cuda_device_id device_id)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};
  auto& map = detail::get_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id.value());
  return (found == map.end()) ? (map[device_id.value()] = detail::initial_resource())
                              : found->second;
}

namespace detail {

// The non-thread-safe implementation of `set_per_device_resource_ref`. This exists because
// we need to call this function from two places: the thread-safe version of
// `set_per_device_resource_ref` and the thread-safe version of `set_per_device_resource`,
// both of which take the lock, so we need an implementation that doesn't take the lock.
/// @private
inline device_async_resource_ref set_per_device_resource_ref_unsafe(
  cuda_device_id device_id, device_async_resource_ref new_resource_ref)
{
  auto& map          = detail::get_ref_map();
  auto const old_itr = map.find(device_id.value());
  // If a resource didn't previously exist for `device_id`, return pointer to initial_resource
  // Note: because resource_ref is not default-constructible, we can't use std::map::operator[]
  if (old_itr == map.end()) {
    map.insert({device_id.value(), new_resource_ref});
    return device_async_resource_ref{detail::initial_resource()};
  }

  auto old_resource_ref = old_itr->second;
  old_itr->second       = new_resource_ref;  // update map directly via iterator
  return old_resource_ref;
}
}  // namespace detail

/**
 * @brief Set the `device_memory_resource` for the specified device.
 *
 * If `new_mr` is not `nullptr`, sets the memory resource pointer for the device specified by `id`
 * to `new_mr`. Otherwise, resets `id`s resource to the initial `cuda_memory_resource`.
 *
 * `id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is undefined.
 *
 * The object pointed to by `new_mr` must outlive the last use of the resource, otherwise behavior
 * is undefined. It is the caller's responsibility to maintain the lifetime of the resource
 * object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @note The resource passed in `new_mr` must have been created when device `id` was the current
 * CUDA device (e.g. set using `cudaSetDevice()`). The behavior of a device_memory_resource is
 * undefined if used while the active CUDA device is a different device from the one that was active
 * when the device_memory_resource was created.
 *
 * @param device_id The id of the target device
 * @param new_mr If not `nullptr`, pointer to new `device_memory_resource` to use as new resource
 * for `id`
 * @return Pointer to the previous memory resource for `id`
 */
inline device_memory_resource* set_per_device_resource(cuda_device_id device_id,
                                                       device_memory_resource* new_mr)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};

  // Note: even though set_per_device_resource() and set_per_device_resource_ref() are not
  // interchangeable, we call the latter from the former to maintain resource_ref
  // state consistent with the resource pointer state. This is necessary because the
  // Python API still uses the raw pointer API. Once the Python API is updated to use
  // resource_ref, this call can be removed.
  detail::set_per_device_resource_ref_unsafe(device_id, new_mr);

  auto& map          = detail::get_map();
  auto const old_itr = map.find(device_id.value());
  // If a resource didn't previously exist for `id`, return pointer to initial_resource
  auto* old_mr           = (old_itr == map.end()) ? detail::initial_resource() : old_itr->second;
  map[device_id.value()] = (new_mr == nullptr) ? detail::initial_resource() : new_mr;
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
 * @note The returned `device_memory_resource` should only be used with the current CUDA device.
 * Changing the current device (e.g. using `cudaSetDevice()`) and then using the returned resource
 * can result in undefined behavior. The behavior of a device_memory_resource is undefined if used
 * while the active CUDA device is a different device from the one that was active when the
 * device_memory_resource was created.
 *
 * @return Pointer to the resource for the current device
 */
inline device_memory_resource* get_current_device_resource()
{
  return get_per_device_resource(rmm::get_current_cuda_device());
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
 * is undefined. It is the caller's responsibility to maintain the lifetime of the resource
 * object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @note The resource passed in `new_mr` must have been created for the current CUDA device. The
 * behavior of a device_memory_resource is undefined if used while the active CUDA device is a
 * different device from the one that was active when the device_memory_resource was created.
 *
 * @param new_mr If not `nullptr`, pointer to new resource to use for the current device
 * @return Pointer to the previous resource for the current device
 */
inline device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)
{
  return set_per_device_resource(rmm::get_current_cuda_device(), new_mr);
}

/**
 * @brief Get the `device_async_resource_ref` for the specified device.
 *
 * Returns a `device_async_resource_ref` for the specified device. The initial resource_ref
 * references a `cuda_memory_resource`.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref`. Concurrent calls to
 * any of these functions will result in a valid state, but the order of execution is undefined.
 *
 * @note The returned `device_async_resource_ref` should only be used when CUDA device `device_id`
 * is the current device  (e.g. set using `cudaSetDevice()`). The behavior of a
 * `device_async_resource_ref` is undefined if used while the active CUDA device is a different
 * device from the one that was active when the memory resource was created.
 *
 * @param device_id The id of the target device
 * @return The current `device_async_resource_ref` for device `device_id`
 */
inline device_async_resource_ref get_per_device_resource_ref(cuda_device_id device_id)
{
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  auto& map = detail::get_ref_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id.value());
  if (found == map.end()) {
    auto item = map.insert({device_id.value(), detail::initial_resource()});
    return item.first->second;
  }
  return found->second;
}

/**
 * @brief Set the `device_async_resource_ref` for the specified device to `new_resource_ref`
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * The object referenced by `new_resource_ref` must outlive the last use of the resource, otherwise
 * behavior is undefined. It is the caller's responsibility to maintain the lifetime of the resource
 * object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref. Concurrent calls to any
 * of these functions will result in a valid state, but the order of execution is undefined.
 *
 * @note The resource passed in `new_resource_ref` must have been created when device `device_id`
 * was the current CUDA device (e.g. set using `cudaSetDevice()`). The behavior of a
 * `device_async_resource_ref` is undefined if used while the active CUDA device is a different
 * device from the one that was active when the memory resource was created.
 *
 * @param device_id The id of the target device
 * @param new_resource_ref new `device_async_resource_ref` to use as new resource for `device_id`
 * @return The previous `device_async_resource_ref` for `device_id`
 */
inline device_async_resource_ref set_per_device_resource_ref(
  cuda_device_id device_id, device_async_resource_ref new_resource_ref)
{
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  return detail::set_per_device_resource_ref_unsafe(device_id, new_resource_ref);
}

/**
 * @brief Get the `device_async_resource_ref` for the current device.
 *
 * Returns the `device_async_resource_ref` set for the current device. The initial resource_ref
 * references a `cuda_memory_resource`.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref. Concurrent calls to any
 * of these functions will result in a valid state, but the order of execution is undefined.

 *
 * @note The returned `device_async_resource_ref` should only be used with the current CUDA device.
 * Changing the current device (e.g. using `cudaSetDevice()`) and then using the returned
 * `resource_ref` can result in undefined behavior. The behavior of a `device_async_resource_ref` is
 * undefined if used while the active CUDA device is a different device from the one that was active
 * when the memory resource was created.
 *
 * @return `device_async_resource_ref` active for the current device
 */
inline device_async_resource_ref get_current_device_resource_ref()
{
  return get_per_device_resource_ref(rmm::get_current_cuda_device());
}

/**
 * @brief Set the `device_async_resource_ref` for the current device.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * The object referenced by `new_resource_ref` must outlive the last use of the resource, otherwise
 * behavior is undefined. It is the caller's responsibility to maintain the lifetime of the resource
 * object.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref. Concurrent calls to any
 * of these functions will result in a valid state, but the order of execution is undefined.
 *
 * @note The resource passed in `new_resource` must have been created for the current CUDA device.
 * The behavior of a `device_async_resource_ref` is undefined if used while the active CUDA device
 * is a different device from the one that was active when the memory resource was created.
 *
 * @param new_resource_ref New `device_async_resource_ref` to use for the current device
 * @return Previous `device_async_resource_ref` for the current device
 */
inline device_async_resource_ref set_current_device_resource_ref(
  device_async_resource_ref new_resource_ref)
{
  return set_per_device_resource_ref(rmm::get_current_cuda_device(), new_resource_ref);
}

/**
 * @brief Reset the `device_async_resource_ref` for the specified device to the initial resource.
 *
 * Resets to a reference to the initial `cuda_memory_resource`.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref. Concurrent calls to any
 * of these functions will result in a valid state, but the order of execution is undefined.
 *
 * @param device_id The id of the target device
 * @return Previous `device_async_resource_ref` for `device_id`
 */
inline device_async_resource_ref reset_per_device_resource_ref(cuda_device_id device_id)
{
  return set_per_device_resource_ref(device_id, detail::initial_resource());
}

/**
 * @brief Reset the `device_async_resource_ref` for the current device to the initial resource.
 *
 * Resets to a reference to the initial `cuda_memory_resource`. The "current device" is the device
 * returned by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource_ref`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource_ref` and `reset_current_device_resource_ref. Concurrent calls to any
 * of these functions will result in a valid state, but the order of execution is undefined.
 *
 * @return Previous `device_async_resource_ref` for `device_id`
 */
inline device_async_resource_ref reset_current_device_resource_ref()
{
  return reset_per_device_resource_ref(rmm::get_current_cuda_device());
}
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
