/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>

namespace RMM_NAMESPACE {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_async_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_resource_ref =
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_async_resource_ref =
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Convert pointer to memory resource into `device_async_resource_ref`, checking for
 * `nullptr`
 *
 * @tparam Resource The type of the memory resource.
 * @param res A pointer to the memory resource.
 * @return A `device_async_resource_ref` to the memory resource.
 * @throws std::logic_error if the memory resource pointer is null.
 */
template <class Resource>
device_async_resource_ref to_device_async_resource_ref_checked(Resource* res)
{
  RMM_EXPECTS(res, "Unexpected null resource pointer.");
  return device_async_resource_ref{*res};
}

// Verify that host_device resource refs can be converted to device-only and host-only resource
// refs. This is needed because a resource that is both host and device accessible can be used in
// contexts that only require one or the other.
static_assert(
  std::is_constructible_v<device_async_resource_ref, host_device_async_resource_ref>,
  "device_async_resource_ref must be constructible from host_device_async_resource_ref");
static_assert(std::is_constructible_v<device_resource_ref, host_device_resource_ref>,
              "device_resource_ref must be constructible from host_device_resource_ref");
static_assert(std::is_constructible_v<host_async_resource_ref, host_device_async_resource_ref>,
              "host_async_resource_ref must be constructible from host_device_async_resource_ref");
static_assert(std::is_constructible_v<host_resource_ref, host_device_resource_ref>,
              "host_resource_ref must be constructible from host_device_resource_ref");

/** @} */  // end of group
}  // namespace RMM_NAMESPACE
