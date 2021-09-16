/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <thrust/execution_policy.h>

namespace rmm {

using par_t         = decltype(thrust::cuda::par(*(new rmm::mr::thrust_allocator<char>())));
using deleter_t     = std::function<void(par_t*)>;
using exec_policy_t = std::unique_ptr<par_t, deleter_t>;

/**
 * @brief Returns a unique_ptr to a Thrust CUDA execution policy that uses RMM
 * for temporary memory allocation.
 *
 * @Param stream The stream that the allocator will use
 *
 * @Returns A Thrust execution policy that will use RMM for temporary memory
 * allocation.
 */
[[deprecated("Use new exec_policy in rmm/exec_policy.hpp")]] inline exec_policy_t exec_policy(
  cudaStream_t stream = nullptr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* alloc  = new rmm::mr::thrust_allocator<char>(cuda_stream_view{stream});
  auto deleter = [alloc](par_t* pointer) {
    delete alloc;    // NOLINT(cppcoreguidelines-owning-memory)
    delete pointer;  // NOLINT(cppcoreguidelines-owning-memory)
  };

  exec_policy_t policy{new par_t(*alloc), deleter};
  return policy;
}

}  // namespace rmm
