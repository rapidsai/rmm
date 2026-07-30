/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>

RMM_NAMESPACE_BEGIN
namespace mr {

cuda_async_managed_memory_resource::cuda_async_managed_memory_resource()
  : shared_base(cuda::mr::make_shared_resource<detail::cuda_async_managed_memory_resource_impl>(
      detail::cuda_async_managed_memory_resource_impl::construction_tag{}))
{
}

cudaMemPool_t cuda_async_managed_memory_resource::pool_handle() const noexcept
{
  return get().pool_handle();
}

}  // namespace mr
RMM_NAMESPACE_END
