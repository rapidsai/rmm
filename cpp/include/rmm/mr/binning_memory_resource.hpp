/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/binning_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <optional>

namespace RMM_EXPORT_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Allocates memory from upstream resources associated with bin sizes.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying bins.
 */
class RMM_EXPORT binning_memory_resource
  : public cuda::mr::shared_resource<detail::binning_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::binning_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `binning_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(binning_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new binning memory resource object.
   *
   * Initially has no bins, so simply uses the upstream resource until bin resources are added
   * with `add_bin`.
   *
   * @param upstream The resource used to allocate bin pools.
   */
  explicit binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);

  /**
   * @brief Construct a new binning memory resource object with a range of initial bins.
   *
   * Constructs a new binning memory resource and adds bins backed by `fixed_size_memory_resource`
   * in the range [2^min_size_exponent, 2^max_size_exponent]. For example if `min_size_exponent==18`
   * and `max_size_exponent==22`, creates bins of sizes 256KiB, 512KiB, 1024KiB, 2048KiB and
   * 4096KiB.
   *
   * @param upstream The resource used to allocate bin pools.
   * @param min_size_exponent The minimum base-2 exponent bin size.
   * @param max_size_exponent The maximum base-2 exponent bin size.
   */
  binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                          int8_t min_size_exponent,  // NOLINT(bugprone-easily-swappable-parameters)
                          int8_t max_size_exponent);

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
  [[nodiscard]] bool operator==(binning_memory_resource const& other) const noexcept;
  /**
   * @brief Compare two resources for inequality.
   *
   * @param other The other resource to compare against
   * @return true if the resources do not compare equal, false otherwise
   */
  [[nodiscard]] bool operator!=(binning_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  ~binning_memory_resource();

  /**
   * @briefreturn{device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Add a bin allocator to this resource
   *
   * Adds `bin_resource` if provided; otherwise constructs and adds a fixed_size_memory_resource.
   *
   * This bin will be used for any allocation smaller than `allocation_size` that is larger than
   * the next smaller bin's allocation size.
   *
   * If there is already a bin of the specified size nothing is changed.
   *
   * This function is not thread safe.
   *
   * @param allocation_size The maximum size that this bin allocates
   * @param bin_resource The memory resource for the bin
   */
  void add_bin(std::size_t allocation_size,
               std::optional<device_async_resource_ref> bin_resource = std::nullopt);
};

static_assert(cuda::mr::resource_with<binning_memory_resource, cuda::mr::device_accessible>,
              "binning_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_EXPORT_NAMESPACE
