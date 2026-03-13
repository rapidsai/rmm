/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/system_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for sam_headroom_memory_resource.
 *
 * Uses a system_memory_resource for allocation with GPU headroom management.
 * This class satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `sam_headroom_memory_resource` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class sam_headroom_memory_resource_impl {
 public:
  explicit sam_headroom_memory_resource_impl(std::size_t headroom);

  ~sam_headroom_memory_resource_impl() = default;

  sam_headroom_memory_resource_impl(sam_headroom_memory_resource_impl const&)            = delete;
  sam_headroom_memory_resource_impl(sam_headroom_memory_resource_impl&&)                 = delete;
  sam_headroom_memory_resource_impl& operator=(sam_headroom_memory_resource_impl const&) = delete;
  sam_headroom_memory_resource_impl& operator=(sam_headroom_memory_resource_impl&&)      = delete;

  [[nodiscard]] bool operator==(sam_headroom_memory_resource_impl const& other) const noexcept
  {
    if (this == std::addressof(other)) { return true; }
    return headroom_ == other.headroom_;
  }

  [[nodiscard]] bool operator!=(sam_headroom_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource_impl const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

 private:
  ///< The system memory resource used for satisfying allocation requests
  system_memory_resource system_mr_;
  ///< Size of GPU memory reserved as headroom
  std::size_t headroom_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
