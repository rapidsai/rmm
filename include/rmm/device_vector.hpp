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

#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <thrust/device_vector.h>

namespace rmm {
/**
 * @addtogroup thrust_integrations
 * @{
 * @file
 */
/**
 * @brief Alias for a thrust::device_vector that uses RMM for memory allocation.
 *
 */
template <typename T>
using device_vector = thrust::device_vector<T, rmm::mr::thrust_allocator<T>>;

/** @} */  // end of group
}  // namespace rmm
