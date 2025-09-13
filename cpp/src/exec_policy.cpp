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

#include <rmm/exec_policy.hpp>

namespace rmm {

exec_policy::exec_policy(cuda_stream_view stream, device_async_resource_ref mr)
  : thrust_exec_policy_t(
      thrust::cuda::par(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
{
}

exec_policy_nosync::exec_policy_nosync(cuda_stream_view stream, device_async_resource_ref mr)
  : thrust_exec_policy_nosync_t(
      thrust::cuda::par_nosync(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
{
}

}  // namespace rmm
