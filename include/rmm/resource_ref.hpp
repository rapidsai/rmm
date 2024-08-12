/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda/memory_resource>

namespace rmm {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_async_resource_ref = cuda::mr::async_resource_ref<cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_async_resource_ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_resource_ref =
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_async_resource_ref =
  cuda::mr::async_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/** @} */  // end of group
}  // namespace rmm
