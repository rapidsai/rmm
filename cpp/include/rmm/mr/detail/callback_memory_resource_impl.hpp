/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <functional>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

/**
 * @brief Implementation class for callback_memory_resource.
 *
 * Holds the allocate/deallocate callbacks and their arguments. This class
 * satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `callback_memory_resource` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class callback_memory_resource_impl {
 public:
  callback_memory_resource_impl(
    std::function<void*(cuda::stream_ref, std::size_t, std::size_t, void*)> allocate_callback,
    std::function<void(cuda::stream_ref, void*, std::size_t, std::size_t, void*)>
      deallocate_callback,
    void* allocate_callback_arg,
    void* deallocate_callback_arg) noexcept;

  ~callback_memory_resource_impl() = default;

  callback_memory_resource_impl(callback_memory_resource_impl const&)            = delete;
  callback_memory_resource_impl(callback_memory_resource_impl&&)                 = delete;
  callback_memory_resource_impl& operator=(callback_memory_resource_impl const&) = delete;
  callback_memory_resource_impl& operator=(callback_memory_resource_impl&&)      = delete;

  bool operator==(callback_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(callback_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = alignof(std::max_align_t));

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(callback_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::function<void*(cuda::stream_ref, std::size_t, std::size_t, void*)> allocate_callback_;
  std::function<void(cuda::stream_ref, void*, std::size_t, std::size_t, void*)>
    deallocate_callback_;
  void* allocate_callback_arg_;
  void* deallocate_callback_arg_;
};

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
