/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <atomic>
#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for limiting_resource_adaptor.
 *
 * Limits the total bytes allocatable through the upstream resource. This class
 * satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `limiting_resource_adaptor` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class limiting_resource_adaptor_impl {
 public:
  limiting_resource_adaptor_impl(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                 std::size_t allocation_limit,
                                 std::size_t alignment);

  ~limiting_resource_adaptor_impl() = default;

  limiting_resource_adaptor_impl(limiting_resource_adaptor_impl const&)            = delete;
  limiting_resource_adaptor_impl(limiting_resource_adaptor_impl&&)                 = delete;
  limiting_resource_adaptor_impl& operator=(limiting_resource_adaptor_impl const&) = delete;
  limiting_resource_adaptor_impl& operator=(limiting_resource_adaptor_impl&&)      = delete;

  bool operator==(limiting_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(limiting_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t get_allocated_bytes() const;

  [[nodiscard]] std::size_t get_allocation_limit() const;

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

  RMM_CONSTEXPR_FRIEND void get_property(limiting_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::size_t allocation_limit_;
  std::atomic<std::size_t> allocated_bytes_;
  std::size_t alignment_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
