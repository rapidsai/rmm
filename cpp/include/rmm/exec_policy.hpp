/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

/**
 @file exec_policy.hpp
 Thrust execution policy that uses RMM's Thrust Allocator Adaptor.
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/version.h>

namespace RMM_NAMESPACE {
/**
 * @addtogroup thrust_integrations
 * @{
 * @file
 */

/**
 * @brief Synchronous execution policy for allocations using Thrust
 */
using thrust_exec_policy_t =
  thrust::detail::execute_with_allocator<mr::thrust_allocator<char>,
                                         thrust::cuda_cub::execute_on_stream_base>;

/**
 * @brief Helper class usable as a Thrust CUDA execution policy
 * that uses RMM for temporary memory allocation on the specified stream.
 */
class exec_policy : public thrust_exec_policy_t {
 public:
  /**
   * @brief Construct a new execution policy object
   *
   * @param stream The stream on which to allocate temporary memory
   * @param mr The resource to use for allocating temporary memory
   */
  explicit exec_policy(cuda_stream_view stream      = cuda_stream_default,
                       device_async_resource_ref mr = mr::get_current_device_resource_ref());
};

/**
 * @brief Asynchronous execution policy for allocations using Thrust
 */
using thrust_exec_policy_nosync_t =
  thrust::detail::execute_with_allocator<mr::thrust_allocator<char>,
                                         thrust::cuda_cub::execute_on_stream_nosync_base>;

/**
 * @brief Helper class usable as a Thrust CUDA execution policy
 * that uses RMM for temporary memory allocation on the specified stream
 * and which allows the Thrust backend to skip stream synchronizations that
 * are not required for correctness.
 */
class exec_policy_nosync : public thrust_exec_policy_nosync_t {
 public:
  /**
   * @brief Construct a new execution policy object
   *
   * @param stream The stream on which to allocate temporary memory
   * @param mr The resource to use for allocating temporary memory
   */
  explicit exec_policy_nosync(cuda_stream_view stream      = cuda_stream_default,
                              device_async_resource_ref mr = mr::get_current_device_resource_ref());
};

/** @} */  // end of group
}  // namespace RMM_NAMESPACE
