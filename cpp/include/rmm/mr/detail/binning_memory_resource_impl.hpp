/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/fixed_size_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for binning_memory_resource.
 *
 * Dispatches allocations to fixed-size bin resources based on size. This class
 * satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `binning_memory_resource` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 */
class binning_memory_resource_impl {
 public:
  explicit binning_memory_resource_impl(device_async_resource_ref upstream);

  binning_memory_resource_impl(device_async_resource_ref upstream,
                               int8_t min_size_exponent,
                               int8_t max_size_exponent);

  ~binning_memory_resource_impl() = default;

  binning_memory_resource_impl(binning_memory_resource_impl const&)            = delete;
  binning_memory_resource_impl(binning_memory_resource_impl&&)                 = delete;
  binning_memory_resource_impl& operator=(binning_memory_resource_impl const&) = delete;
  binning_memory_resource_impl& operator=(binning_memory_resource_impl&&)      = delete;

  bool operator==(binning_memory_resource_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(binning_memory_resource_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  void add_bin(std::size_t allocation_size,
               std::optional<device_async_resource_ref> bin_resource = std::nullopt);

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

  RMM_CONSTEXPR_FRIEND void get_property(binning_memory_resource_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  [[nodiscard]] device_async_resource_ref get_resource_ref(std::size_t bytes);

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  std::vector<std::unique_ptr<fixed_size_memory_resource>> owned_bin_resources_;
  std::map<std::size_t, device_async_resource_ref> resource_bins_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
