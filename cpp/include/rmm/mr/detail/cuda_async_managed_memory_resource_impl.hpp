/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for cuda_async_managed_memory_resource.
 *
 * Uses the default managed memory pool via `cudaMemGetDefaultMemPool` with
 * `cudaMemAllocationTypeManaged`. This class satisfies the CCCL
 * `cuda::mr::resource` concept and is held by
 * `cuda_async_managed_memory_resource` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class cuda_async_managed_memory_resource_impl {
 public:
  cuda_async_managed_memory_resource_impl();

  ~cuda_async_managed_memory_resource_impl() = default;

  cuda_async_managed_memory_resource_impl(cuda_async_managed_memory_resource_impl const&) = delete;
  cuda_async_managed_memory_resource_impl(cuda_async_managed_memory_resource_impl&&)      = delete;
  cuda_async_managed_memory_resource_impl& operator=(
    cuda_async_managed_memory_resource_impl const&) = delete;
  cuda_async_managed_memory_resource_impl& operator=(cuda_async_managed_memory_resource_impl&&) =
    delete;

  [[nodiscard]] bool operator==(cuda_async_managed_memory_resource_impl const& other) const noexcept
  {
    if (this == std::addressof(other)) { return true; }
    return pool_handle() == other.pool_handle();
  }

  [[nodiscard]] bool operator!=(cuda_async_managed_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @brief Returns the underlying native handle to the CUDA pool
   *
   * @return cudaMemPool_t Handle to the underlying CUDA pool
   */
  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  RMM_CONSTEXPR_FRIEND void get_property(cuda_async_managed_memory_resource_impl const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

 private:
  cuda_async_view_memory_resource pool_{};
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
