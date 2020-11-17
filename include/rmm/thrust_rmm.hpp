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
 @file thrust_rmm.hpp
 Allocator class compatible with thrust arrays that uses RMM device memory
 manager.
 */

#ifndef THRUST_RMM_HPP
#define THRUST_RMM_HPP

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace rmm {

/**
 * @brief Returns a Thrust CUDA execution policy that uses RMM for temporary memory allocation.
 */
inline auto exec_policy(cuda_stream_view stream = cuda_stream_default)
{
  return thrust::cuda::par(rmm::mr::thrust_allocator<char>(stream)).on(stream.value());
}

}  // namespace rmm

#endif  // THRUST_RMM_HPP
