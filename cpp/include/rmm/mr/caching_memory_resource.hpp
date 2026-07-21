/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/caching_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <optional>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A stream-ordered caching suballocator with separate small and large allocation policies.
 *
 * This resource is modeled after the broad design of PyTorch's CUDA caching allocator. It
 * allocates larger segments from an upstream resource, then suballocates from cached free blocks
 * using best fit with coalescing.
 *
 * Allocation and deallocation are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * Requests are rounded to upstream segment sizes using the following policy:
 * - requests <= 1 MiB allocate from 2 MiB upstream segments
 * - requests in (1 MiB, 10 MiB) allocate from 20 MiB upstream segments
 * - requests >= 10 MiB allocate from upstream sizes rounded up to 2 MiB
 *
 * When an upstream allocation fails while growing the cache, the resource releases cached whole
 * segments from the relevant pool and retries once.
 *
 * This class is copyable and shares ownership of its internal state, allowing multiple instances
 * to safely reference the same underlying cache.
 */
class RMM_EXPORT caching_memory_resource
  : public cuda::mr::shared_resource<detail::caching_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::caching_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property.
   */
  RMM_CONSTEXPR_FRIEND void get_property(caching_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a `caching_memory_resource`.
   *
   * @param upstream The resource from which cached segments are allocated.
   * @param max_split_size Optional threshold for the large-allocation pool. If set, large cached
   * blocks at or above this size will not be split to satisfy smaller requests.
   * @param oom_fallback_policy Policy used after an upstream allocation failure.
   * @param pool_policy Policy controlling cross-pool cached block reuse.
   * @param stream_reuse_policy Policy controlling cross-stream cached block reuse.
   */
  explicit caching_memory_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::optional<std::size_t> max_split_size = std::nullopt,
    caching_memory_resource_oom_fallback_policy oom_fallback_policy =
      caching_memory_resource_oom_fallback_policy::release_oversized_then_all,
    caching_memory_resource_pool_policy pool_policy = caching_memory_resource_pool_policy::separate,
    caching_memory_resource_stream_reuse_policy stream_reuse_policy =
      caching_memory_resource_stream_reuse_policy::same_stream);

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Release all cached whole segments.
   *
   * Outstanding allocations are not affected.
   *
   * @return Number of bytes returned to the upstream resource.
   */
  [[nodiscard]] std::size_t release() noexcept;

  /**
   * @brief Release cached whole segments from the small-allocation pool.
   *
   * @return Number of bytes returned to the upstream resource.
   */
  [[nodiscard]] std::size_t release_small_blocks() noexcept;

  /**
   * @brief Release cached whole segments from the large-allocation pool.
   *
   * @return Number of bytes returned to the upstream resource.
   */
  [[nodiscard]] std::size_t release_large_blocks() noexcept;

  /**
   * @brief Return the total number of cached bytes currently owned by the small-allocation pool.
   *
   * @return Total cached bytes in the small-allocation pool.
   */
  [[nodiscard]] std::size_t cached_small_bytes() const noexcept;

  /**
   * @brief Return the total number of cached bytes currently owned by the large-allocation pool.
   *
   * @return Total cached bytes in the large-allocation pool.
   */
  [[nodiscard]] std::size_t cached_large_bytes() const noexcept;

  /**
   * @brief Return the upstream segment size used for a request of `bytes` bytes.
   *
   * This exposes the allocator's request-rounding policy.
   *
   * @param bytes The requested allocation size in bytes.
   * @return Upstream segment size used for the request.
   */
  [[nodiscard]] std::size_t upstream_allocation_size(std::size_t bytes) const noexcept;
};

static_assert(cuda::mr::resource_with<caching_memory_resource, cuda::mr::device_accessible>,
              "caching_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group

}  // namespace mr
}  // namespace RMM_NAMESPACE
