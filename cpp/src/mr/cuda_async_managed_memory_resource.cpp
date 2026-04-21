/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

cuda_async_managed_memory_resource::cuda_async_managed_memory_resource()
  : shared_base(cuda::mr::make_shared_resource<detail::cuda_async_managed_memory_resource_impl>())
{
}

cudaMemPool_t cuda_async_managed_memory_resource::pool_handle() const noexcept
{
  return get().pool_handle();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
