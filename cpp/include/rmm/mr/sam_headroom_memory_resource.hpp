/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/sam_headroom_memory_resource_impl.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

namespace RMM_EXPORT_NAMESPACE {
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

  /**
   * @brief Allocate memory using this resource.
   *
   * @param stream Stream on which to perform the allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory using this resource.
   *
   * @param stream Stream on which to perform deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Allocate memory synchronously using this resource.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory synchronously using this resource.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the allocation call
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Compare two resources for equality.
   *
   * @param other The other resource to compare against
   * @return true if the resources compare equal, false otherwise
   */
  [[nodiscard]] bool operator==(sam_headroom_memory_resource const& other) const noexcept;
  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other resource to compare against
   * @return true if the resources do not compare equal, false otherwise
   */
  [[nodiscard]] bool operator!=(sam_headroom_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  sam_headroom_memory_resource() = delete;
  ~sam_headroom_memory_resource();
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
}  // namespace RMM_EXPORT_NAMESPACE
