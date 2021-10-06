/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <thrust/system/cuda/execution_policy.h>

namespace rmm {

using thrust_exec_policy_t =
  thrust::detail::execute_with_allocator<rmm::mr::thrust_allocator<char>,
                                         thrust::cuda_cub::execute_on_stream_base>;

/**
 * @brief Helper class usable as a Thrust CUDA execution policy
 * that uses RMM for temporary memory allocation on the specified stream.
 */
class exec_policy : public thrust_exec_policy_t {
 public:
  explicit exec_policy(cuda_stream_view stream             = cuda_stream_default,
                       rmm::mr::device_memory_resource* mr = mr::get_current_device_resource())
    : thrust_exec_policy_t(
        thrust::cuda::par(rmm::mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
  {
  }
};

}  // namespace rmm
