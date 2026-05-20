/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/exec_policy.hpp>

namespace rmm {

exec_policy::exec_policy(cuda_stream_view stream,
                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : thrust_exec_policy_t(thrust::cuda::par(mr::thrust_allocator<char>(stream, std::move(mr)))
                           .on(cuda::stream_ref{stream}.get()))
{
}

exec_policy_nosync::exec_policy_nosync(cuda_stream_view stream,
                                       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : thrust_exec_policy_nosync_t(
      thrust::cuda::par_nosync(mr::thrust_allocator<char>(stream, std::move(mr)))
        .on(cuda::stream_ref{stream}.get()))
{
}

}  // namespace rmm
