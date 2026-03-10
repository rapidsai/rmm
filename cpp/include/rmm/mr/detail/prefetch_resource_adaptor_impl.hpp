/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for prefetch_resource_adaptor.
 *
 * Prefetches all allocations to the current device. This class satisfies the
 * CCCL `cuda::mr::resource` concept and is held by `prefetch_resource_adaptor`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class prefetch_resource_adaptor_impl {
 public:
  explicit prefetch_resource_adaptor_impl(device_async_resource_ref upstream);

  ~prefetch_resource_adaptor_impl() = default;

  prefetch_resource_adaptor_impl(prefetch_resource_adaptor_impl const&)            = delete;
  prefetch_resource_adaptor_impl(prefetch_resource_adaptor_impl&&)                 = delete;
  prefetch_resource_adaptor_impl& operator=(prefetch_resource_adaptor_impl const&) = delete;
  prefetch_resource_adaptor_impl& operator=(prefetch_resource_adaptor_impl&&)      = delete;

  bool operator==(prefetch_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(prefetch_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(prefetch_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
