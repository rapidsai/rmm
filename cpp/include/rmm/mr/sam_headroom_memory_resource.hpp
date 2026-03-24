/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/sam_headroom_memory_resource_impl.hpp>

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
  : public cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::sam_headroom_memory_resource_impl>;

 public:
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

  sam_headroom_memory_resource()  = delete;
  ~sam_headroom_memory_resource() = default;
  sam_headroom_memory_resource(sam_headroom_memory_resource const&) =
    default;  ///< @default_copy_constructor
  sam_headroom_memory_resource(sam_headroom_memory_resource&&) =
    default;  ///< @default_move_constructor
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_headroom_memory_resource}
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_headroom_memory_resource}
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
