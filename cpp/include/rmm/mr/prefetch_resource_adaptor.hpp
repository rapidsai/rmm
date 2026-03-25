/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/prefetch_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that prefetches all memory allocations.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT prefetch_resource_adaptor
  : public cuda::mr::shared_resource<detail::prefetch_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::prefetch_resource_adaptor_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(prefetch_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new prefetch resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream The resource_ref used for allocating/deallocating device memory
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, prefetch_resource_adaptor>, int> = 0>
  explicit prefetch_resource_adaptor(Upstream&& upstream)
    : shared_base(cuda::mr::make_shared_resource<detail::prefetch_resource_adaptor_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream)}))
  {
  }

  ~prefetch_resource_adaptor() = default;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;
};

static_assert(cuda::mr::resource_with<prefetch_resource_adaptor, cuda::mr::device_accessible>,
              "prefetch_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
