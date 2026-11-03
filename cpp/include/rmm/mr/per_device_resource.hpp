/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <map>
#include <mutex>
#include <utility>

/**
 * @file per_device_resource.hpp
 * @brief Management of per-device memory resources
 *
 * Provides functions to get/set the default `device_async_resource_ref` for each CUDA device. The
 * initial resource is a `cuda_async_memory_resource` when stream-ordered allocation is supported
 * and a `cuda_memory_resource` otherwise.
 *
 * @note Memory resources make CUDA API calls without setting the current CUDA device.
 * Therefore a memory resource should only be used with the current CUDA device set to the device
 * that was active when the memory resource was created.
 */

RMM_NAMESPACE_BEGIN
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
 * @brief Owning type-erased device resource stored for each device.
 */
using owning_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

/**
 * @brief Initial and current resources for each device, guarded by a single lock.
 */
struct per_device_resources {
  std::mutex lock;  ///< Guards `initial` and `current`
  std::map<cuda_device_id::value_type, owning_resource>
    initial;  ///< Map from device id -> initial resource
  std::map<cuda_device_id::value_type, owning_resource>
    current;  ///< Map from device id -> current resource
};

/**
 * @briefreturn{Reference to the process-wide per-device resources}
 */
RMM_EXPORT inline per_device_resources& get_per_device_resources()
{
  static per_device_resources resources;
  // Register the process-exit hook immediately after constructing the map.
  rmm::detail::register_process_exit_hook();
  return resources;
}

/**
 * @briefreturn{Reference to the lock}
 */
RMM_EXPORT inline std::mutex& ref_map_lock() { return get_per_device_resources().lock; }

// This symbol must have default visibility, see: https://github.com/rapidsai/rmm/issues/826
/**
 * @briefreturn{Reference to the map from device id -> any_resource}
 */
RMM_EXPORT inline auto& get_ref_map() { return get_per_device_resources().current; }

/**
 * @brief Returns a reference to the initial resource for the specified device, constructing it on
 * first use.
 *
 * The caller must hold `ref_map_lock()`.
 *
 * @param device_id The id of the target device
 * @return Reference to the initial resource for `device_id`
 */
inline owning_resource& initial_resource_unlocked(cuda_device_id device_id)
{
  auto& initial    = get_per_device_resources().initial;
  auto const found = initial.find(device_id.value());
  if (found != initial.end()) { return found->second; }

  cuda_set_device_raii set_device{device_id};
  if (rmm::detail::runtime_async_alloc::is_supported()) {
    return initial.emplace(device_id.value(), cuda_async_memory_resource{}).first->second;
  }
  return initial.emplace(device_id.value(), cuda_memory_resource{}).first->second;
}

/**
 * @brief Returns a reference to the initial resource for the specified device.
 *
 * @param device_id The id of the target device
 * @return Reference to the initial resource for `device_id`
 */
RMM_EXPORT inline owning_resource& initial_resource(cuda_device_id device_id)
{
  std::lock_guard lock{ref_map_lock()};
  return initial_resource_unlocked(device_id);
}

/**
 * @brief Returns a reference to the initial resource for the current device.
 *
 * @return Reference to the initial resource for the current device
 */
RMM_EXPORT inline owning_resource& initial_resource()
{
  return initial_resource(rmm::get_current_cuda_device());
}

}  // namespace detail

/**
 * @brief Get the `device_async_resource_ref` for the specified device.
 *
 * Returns a `device_async_resource_ref` for the specified device. The initial resource_ref
 * references a `cuda_async_memory_resource` when supported and a `cuda_memory_resource` otherwise.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.
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
    auto item = map.emplace(device_id.value(), detail::initial_resource_unlocked(device_id));
    return device_async_resource_ref{item.first->second};
  }
  return device_async_resource_ref{found->second};
}

/**
 * @brief Set the memory resource for the specified device.
 *
 * Takes ownership of the provided resource by value. The resource is moved into the per-device
 * resource map.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.
 *
 * @note The resource passed in `new_resource` must have been created when device `device_id`
 * was the current CUDA device (e.g. set using `cudaSetDevice()`). The behavior of a memory
 * resource is undefined if used while the active CUDA device is a different device from the one
 * that was active when the memory resource was created.
 *
 * @note The per-device resource map keeps the provided resource alive until process exit. Its
 * destructor may therefore run during process termination. If the destructor may call CUDA APIs,
 * it must consult `rmm::process_is_exiting()` and skip those calls when it returns `true`.
 *
 * @param device_id The id of the target device
 * @param new_resource New resource to use for `device_id`
 * @return An owning `any_resource` holding the previous resource for `device_id`
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_per_device_resource(
  cuda_device_id device_id, cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
{
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  auto& map          = detail::get_ref_map();
  auto const old_itr = map.find(device_id.value());
  if (old_itr == map.end()) {
    map.emplace(device_id.value(), std::move(new_resource));
    return {detail::initial_resource_unlocked(device_id)};
  }
  return std::exchange(old_itr->second, std::move(new_resource));
}

/**
 * @brief Get the `device_async_resource_ref` for the current device.
 *
 * Returns the `device_async_resource_ref` set for the current device. The initial resource_ref
 * references a `cuda_async_memory_resource` when supported and a `cuda_memory_resource` otherwise.
 *
 * The "current device" is the device returned by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.

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
 * @brief Set the memory resource for the current device.
 *
 * Takes ownership of the provided resource by value. The "current device" is the device returned
 * by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.
 *
 * @note The resource passed in `new_resource` must have been created for the current CUDA device.
 * The behavior of a memory resource is undefined if used while the active CUDA device is a
 * different device from the one that was active when the memory resource was created.
 *
 * @note The per-device resource map keeps the provided resource alive until process exit. Its
 * destructor may therefore run during process termination. If the destructor may call CUDA APIs,
 * it must consult `rmm::process_is_exiting()` and skip those calls when it returns `true`.
 *
 * @param new_resource New resource to use for the current device
 * @return An owning `any_resource` holding the previous resource for the current device
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> new_resource)
{
  return set_per_device_resource(rmm::get_current_cuda_device(), std::move(new_resource));
}

/**
 * @brief Reset the memory resource for the specified device to the initial resource.
 *
 * Resets to the initial resource selected for the specified device.
 *
 * `device_id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is
 * undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.
 *
 * @param device_id The id of the target device
 * @return An owning `any_resource` holding the previous resource for `device_id`
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_per_device_resource(
  cuda_device_id device_id)
{
  return set_per_device_resource(device_id, {detail::initial_resource(device_id)});
}

/**
 * @brief Reset the memory resource for the current device to the initial resource.
 *
 * Resets to the initial resource selected for the current device. The "current device" is the
 * device returned by `cudaGetDevice`.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource_ref`, `get_current_device_resource_ref`,
 * `set_current_device_resource`, `reset_per_device_resource`, and
 * `reset_current_device_resource`. Concurrent calls to any of these functions will result in a
 * valid state, but the order of execution is undefined.
 *
 * @return An owning `any_resource` holding the previous resource for the current device
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource()
{
  return reset_per_device_resource(rmm::get_current_cuda_device());
}

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
