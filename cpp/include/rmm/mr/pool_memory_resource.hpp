/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/pool_memory_resource_impl.hpp>
#include <rmm/mr/device_memory_resource.hpp>
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
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation and deallocation are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying pool.
 */
class RMM_EXPORT pool_memory_resource
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::pool_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::pool_memory_resource_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Equality comparison operator.
   *
   * @param other The other pool_memory_resource to compare against.
   * @return true if both resources share the same underlying state.
   */
  [[nodiscard]] bool operator==(pool_memory_resource const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Inequality comparison operator.
   *
   * @param other The other pool_memory_resource to compare against.
   * @return true if the resources do not share the same underlying state.
   */
  [[nodiscard]] bool operator!=(pool_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pool_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream_mr`.
   *
   * @throws rmm::logic_error if `initial_pool_size` is not aligned to a multiple of 256 bytes.
   * @throws rmm::logic_error if `maximum_pool_size` is neither the default nor aligned to a
   * multiple of 256 bytes.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all
   * of the available memory from the upstream resource.
   */
  explicit pool_memory_resource(device_async_resource_ref upstream_mr,
                                std::size_t initial_pool_size,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt);

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return std::size_t The total size of the currently allocated pool.
   */
  [[nodiscard]] std::size_t pool_size() const noexcept;

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<pool_memory_resource, cuda::mr::device_accessible>,
              "pool_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
