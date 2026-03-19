/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/binning_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
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
   * Initially has no bins, so simply uses the upstream_resource until bin resources are added
   * with `add_bin`.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, binning_memory_resource>, int> = 0>
  explicit binning_memory_resource(Upstream&& upstream_resource)
    : shared_base(cuda::mr::make_shared_resource<detail::binning_memory_resource_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{
          std::forward<Upstream>(upstream_resource)}))
  {
  }

  /**
   * @brief Construct a new binning memory resource object with a range of initial bins.
   *
   * Constructs a new binning memory resource and adds bins backed by `fixed_size_memory_resource`
   * in the range [2^min_size_exponent, 2^max_size_exponent]. For example if `min_size_exponent==18`
   * and `max_size_exponent==22`, creates bins of sizes 256KiB, 512KiB, 1024KiB, 2048KiB and
   * 4096KiB.
   *
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   * @param min_size_exponent The minimum base-2 exponent bin size.
   * @param max_size_exponent The maximum base-2 exponent bin size.
   */
  binning_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream_resource,
                          int8_t min_size_exponent,  // NOLINT(bugprone-easily-swappable-parameters)
                          int8_t max_size_exponent);

  ~binning_memory_resource() = default;

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
}  // namespace RMM_NAMESPACE
