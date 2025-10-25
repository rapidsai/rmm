/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/cccl_adaptors.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

namespace RMM_NAMESPACE {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_resource_ref =
  detail::cccl_resource_ref<cuda::mr::synchronous_resource_ref<cuda::mr::device_accessible>>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_async_resource_ref =
  detail::cccl_async_resource_ref<cuda::mr::resource_ref<cuda::mr::device_accessible>>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_resource_ref =
  detail::cccl_resource_ref<cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_async_resource_ref =
  detail::cccl_async_resource_ref<cuda::mr::resource_ref<cuda::mr::host_accessible>>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_resource_ref = detail::cccl_resource_ref<
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_async_resource_ref = detail::cccl_async_resource_ref<
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>>;

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

/** @} */  // end of group
}  // namespace RMM_NAMESPACE
