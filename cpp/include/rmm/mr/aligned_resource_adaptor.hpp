/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/aligned_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

namespace RMM_EXPORT_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that adapts an upstream resource to allocate memory with a specified alignment.
 *
 * If the requested alignment is smaller than `CUDA_ALLOCATION_ALIGNMENT` (256 bytes) it is
 * increased to `CUDA_ALLOCATION_ALIGNMENT`. An optional threshold controls the minimum size above
 * which the custom alignment is applied.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT aligned_resource_adaptor
  : public cuda::mr::shared_resource<detail::aligned_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::aligned_resource_adaptor_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(aligned_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief The default alignment threshold used by the adaptor (0 = always align).
   */
  static constexpr std::size_t default_alignment_threshold =
    detail::aligned_resource_adaptor_impl::default_alignment_threshold;

  /**
   * @brief Construct an aligned resource adaptor using `upstream` to satisfy allocation requests.
   *
   * @throws rmm::logic_error if `alignment` is not a power of 2
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param alignment The size used for allocation alignment (raised to CUDA_ALLOCATION_ALIGNMENT
   * if smaller).
   * @param alignment_threshold Only allocations >= this size are aligned to `alignment`.
   */
  explicit aligned_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT,
                                    std::size_t alignment_threshold = default_alignment_threshold);

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
  [[nodiscard]] bool operator==(aligned_resource_adaptor const& other) const noexcept;
  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other resource to compare against
   * @return true if the resources do not compare equal, false otherwise
   */
  [[nodiscard]] bool operator!=(aligned_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

  ~aligned_resource_adaptor();

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;
};

static_assert(cuda::mr::resource_with<aligned_resource_adaptor, cuda::mr::device_accessible>,
              "aligned_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_EXPORT_NAMESPACE
