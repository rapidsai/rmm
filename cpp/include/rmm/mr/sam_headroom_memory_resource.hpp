/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/sam_headroom_memory_resource_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */
/**
 * @brief Resource that uses system memory resource to allocate memory with a headroom.
 *
 * System allocated memory (SAM) can be migrated to the GPU, but is never migrated back the host. If
 * GPU memory is over-subscribed, this can cause other CUDA calls to fail with out-of-memory errors.
 * To work around this problem, when using a system memory resource, we reserve some GPU memory as
 * headroom for other CUDA calls, and only conditionally set its preferred location to the GPU if
 * the allocation would not eat into the headroom.
 *
 * Since doing this check on every allocation can be expensive, the caller may choose to use other
 * allocators (e.g. `binning_memory_resource`) for small allocations, and use this allocator for
 * large allocations only.
 */
class RMM_EXPORT sam_headroom_memory_resource final
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Compare two resources for equality (shared-impl identity).
   *
   * @param other The other sam_headroom_memory_resource to compare against.
   * @return true if both resources share the same underlying state.
   */
  [[nodiscard]] bool operator==(sam_headroom_memory_resource const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other sam_headroom_memory_resource to compare against.
   * @return true if the resources do not share the same underlying state.
   */
  [[nodiscard]] bool operator!=(sam_headroom_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(sam_headroom_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Construct a headroom memory resource.
   *
   * @param headroom Size of the reserved GPU memory as headroom
   */
  explicit sam_headroom_memory_resource(std::size_t headroom);

  sam_headroom_memory_resource()                                               = delete;
  ~sam_headroom_memory_resource()                                              = default;
  sam_headroom_memory_resource(sam_headroom_memory_resource const&)            = delete;
  sam_headroom_memory_resource(sam_headroom_memory_resource&&)                 = delete;
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource const&) = delete;
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource&&)      = delete;

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

// static property checks
static_assert(cuda::mr::synchronous_resource<sam_headroom_memory_resource>);
static_assert(cuda::mr::resource<sam_headroom_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<sam_headroom_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<sam_headroom_memory_resource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<sam_headroom_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<sam_headroom_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
