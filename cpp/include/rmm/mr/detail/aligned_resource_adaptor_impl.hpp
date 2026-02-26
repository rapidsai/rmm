/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for aligned_resource_adaptor.
 *
 * Allocates with a specified alignment size. This class satisfies the CCCL
 * `cuda::mr::resource` concept and is held by `aligned_resource_adaptor` via
 * `cuda::mr::shared_resource` for reference-counted ownership.
 */
class aligned_resource_adaptor_impl {
 public:
  static constexpr std::size_t default_alignment_threshold = 0;

  aligned_resource_adaptor_impl(device_async_resource_ref upstream,
                                std::size_t alignment,
                                std::size_t alignment_threshold);

  ~aligned_resource_adaptor_impl() = default;

  aligned_resource_adaptor_impl(aligned_resource_adaptor_impl const&)            = delete;
  aligned_resource_adaptor_impl(aligned_resource_adaptor_impl&&)                 = delete;
  aligned_resource_adaptor_impl& operator=(aligned_resource_adaptor_impl const&) = delete;
  aligned_resource_adaptor_impl& operator=(aligned_resource_adaptor_impl&&)      = delete;

  bool operator==(aligned_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(aligned_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] std::size_t get_alignment() const noexcept;

  [[nodiscard]] std::size_t get_alignment_threshold() const noexcept;

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

  RMM_CONSTEXPR_FRIEND void get_property(aligned_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  [[nodiscard]] std::size_t upstream_allocation_size(std::size_t bytes) const;

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::unordered_map<void*, void*> pointers_;
  std::size_t alignment_;
  std::size_t alignment_threshold_;
  mutable std::mutex mtx_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
