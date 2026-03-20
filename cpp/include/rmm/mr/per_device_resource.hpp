/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/export.hpp>
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
 * Provides functions to get/set the default `device_async_resource_ref` for each CUDA device.
 * The initial resource for each device is a `cuda_memory_resource`.
 *
 * @note Memory resources make CUDA API calls without setting the current CUDA device.
 * Therefore a memory resource should only be used with the current CUDA device set to the device
 * that was active when the memory resource was created.
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
 * @brief Returns a reference to the initial resource.
 *
 * Returns a global instance of a `cuda_memory_resource` as a function local static.
 *
 * @return Reference to the static cuda_memory_resource used as the initial, default resource
 */
RMM_EXPORT inline cuda_memory_resource& initial_resource()
{
  static cuda_memory_resource mr{};
  return mr;
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
 * @briefreturn{Reference to the map from device id -> any_resource}
 */
RMM_EXPORT inline auto& get_ref_map()
{
  static std::map<cuda_device_id::value_type, cuda::mr::any_resource<cuda::mr::device_accessible>>
    device_id_to_resource;
  return device_id_to_resource;
}

}  // namespace detail

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
  using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  auto& map = detail::get_ref_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id.value());
  if (found == map.end()) {
    device_async_resource_ref initial_ref{detail::initial_resource()};
    auto item = map.emplace(device_id.value(), any_device_resource{initial_ref});
    return device_async_resource_ref{item.first->second};
  }
  return device_async_resource_ref{found->second};
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
 * @return An owning `any_resource` holding the previous resource for `device_id`
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_per_device_resource_ref(
  cuda_device_id device_id, device_async_resource_ref new_resource_ref)
{
  using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  auto& map          = detail::get_ref_map();
  auto const old_itr = map.find(device_id.value());
  if (old_itr == map.end()) {
    map.emplace(device_id.value(), any_device_resource{new_resource_ref});
    return any_device_resource{device_async_resource_ref{detail::initial_resource()}};
  }
  return std::exchange(old_itr->second, any_device_resource{new_resource_ref});
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
 * @return An owning `any_resource` holding the previous resource for the current device
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource_ref(
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
 * @return An owning `any_resource` holding the previous resource for `device_id`
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_per_device_resource_ref(
  cuda_device_id device_id)
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
 * @return An owning `any_resource` holding the previous resource for the current device
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource_ref()
{
  return reset_per_device_resource_ref(rmm::get_current_cuda_device());
}
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
