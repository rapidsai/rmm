/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/aligned_resource_adaptor_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace RMM_NAMESPACE {
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
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::aligned_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::aligned_resource_adaptor_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Compare two adaptors for equality (shared-impl identity).
   *
   * @param other The other adaptor to compare against.
   * @return true if both adaptors share the same underlying impl.
   */
  [[nodiscard]] bool operator==(aligned_resource_adaptor const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Compare two adaptors for inequality.
   *
   * @param other The other adaptor to compare against.
   * @return true if the adaptors do not share the same underlying impl.
   */
  [[nodiscard]] bool operator!=(aligned_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

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
  explicit aligned_resource_adaptor(device_async_resource_ref upstream,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT,
                                    std::size_t alignment_threshold = default_alignment_threshold);

  ~aligned_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<aligned_resource_adaptor, cuda::mr::device_accessible>,
              "aligned_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
