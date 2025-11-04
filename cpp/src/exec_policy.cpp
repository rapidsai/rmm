/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
