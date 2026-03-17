/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

cuda_async_memory_resource::cuda_async_memory_resource(
  std::optional<std::size_t> initial_pool_size,
  std::optional<std::size_t> release_threshold,
  std::optional<allocation_handle_type> export_handle_type)
  : shared_base(cuda::mr::make_shared_resource<detail::cuda_async_memory_resource_impl>(
      initial_pool_size,
      release_threshold,
      export_handle_type.has_value()
        ? std::optional<std::int32_t>{static_cast<std::int32_t>(export_handle_type.value())}
        : std::nullopt,
      rmm::detail::hwdecompress::is_supported()))
{
}

cudaMemPool_t cuda_async_memory_resource::pool_handle() const noexcept
{
  return get().pool_handle();
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
