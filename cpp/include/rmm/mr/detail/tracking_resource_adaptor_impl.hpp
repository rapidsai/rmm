/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/detail/stack_trace.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for tracking_resource_adaptor.
 *
 * Tracks allocations made through the upstream resource. This class satisfies
 * the CCCL `cuda::mr::resource` concept and is held by `tracking_resource_adaptor`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class tracking_resource_adaptor_impl {
 public:
  using read_lock_t  = std::shared_lock<std::shared_mutex>;
  using write_lock_t = std::unique_lock<std::shared_mutex>;

  struct allocation_info {
    std::unique_ptr<rmm::detail::stack_trace> strace;
    std::size_t allocation_size;

    allocation_info() = delete;
    allocation_info(std::size_t size, bool capture_stack)
      : strace{capture_stack ? std::make_unique<rmm::detail::stack_trace>() : nullptr},
        allocation_size{size}
    {
    }
  };

  tracking_resource_adaptor_impl(device_async_resource_ref upstream, bool capture_stacks);

  ~tracking_resource_adaptor_impl() = default;

  tracking_resource_adaptor_impl(tracking_resource_adaptor_impl const&)            = delete;
  tracking_resource_adaptor_impl(tracking_resource_adaptor_impl&&)                 = delete;
  tracking_resource_adaptor_impl& operator=(tracking_resource_adaptor_impl const&) = delete;
  tracking_resource_adaptor_impl& operator=(tracking_resource_adaptor_impl&&)      = delete;

  bool operator==(tracking_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(tracking_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::map<void*, allocation_info> const& get_outstanding_allocations()
    const noexcept;

  [[nodiscard]] std::size_t get_allocated_bytes() const noexcept;

  [[nodiscard]] std::string get_outstanding_allocations_str() const;

  void log_outstanding_allocations() const;

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

  RMM_CONSTEXPR_FRIEND void get_property(tracking_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  bool capture_stacks_;
  std::map<void*, allocation_info> allocations_;
  std::atomic<std::size_t> allocated_bytes_{0};
  mutable std::shared_mutex mtx_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
