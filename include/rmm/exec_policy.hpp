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
 * @brief Returns a Thrust CUDA execution policy that uses RMM for temporary memory allocation on
 * the specified stream.
 */
inline auto create_exec_policy(
  cuda_stream_view stream             = cuda_stream_default,
  rmm::mr::device_memory_resource* mr = mr::get_current_device_resource())
{
  return thrust::cuda::par(rmm::mr::thrust_allocator<char>(stream, mr)).on(stream.value());
}

class exec_policy : public thrust_exec_policy_t {
 public:
  exec_policy(cuda_stream_view stream             = cuda_stream_default,
              rmm::mr::device_memory_resource* mr = mr::get_current_device_resource())
    : thrust_exec_policy_t(create_exec_policy(stream, mr))
  {
  }
};

}  // namespace rmm
