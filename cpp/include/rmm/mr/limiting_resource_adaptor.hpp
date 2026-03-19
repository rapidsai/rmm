/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/limiting_resource_adaptor_impl.hpp>
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
 * @brief Resource that uses an upstream resource to allocate memory and limits the total
 * allocations possible.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests, but any existing allocations
 * will be untracked. Atomics are used to make this thread-safe, but note that
 * the `get_allocated_bytes` may not include in-flight allocations.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT limiting_resource_adaptor
  : public cuda::mr::shared_resource<detail::limiting_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::limiting_resource_adaptor_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(limiting_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a new limiting resource adaptor using `upstream` to satisfy
   * allocation requests and limiting the total allocation amount possible.
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param allocation_limit Maximum memory allowed for this allocator
   * @param alignment Alignment in bytes for the start of each allocated buffer
   */
  limiting_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                            std::size_t allocation_limit,
                            std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT);

  ~limiting_resource_adaptor() = default;

  /**
   * @briefreturn{device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Query the number of bytes that have been allocated. Note that
   * this can not be used to know how large of an allocation is possible due
   * to both possible fragmentation and also internal page sizes and alignment
   * that is not tracked by this allocator.
   *
   * @return std::size_t number of bytes that have been allocated through this
   * allocator.
   */
  [[nodiscard]] std::size_t get_allocated_bytes() const;

  /**
   * @brief Query the maximum number of bytes that this allocator is allowed
   * to allocate. This is the limit on the allocator and not a representation of
   * the underlying device. The device may not be able to support this limit.
   *
   * @return std::size_t max number of bytes allowed for this allocator
   */
  [[nodiscard]] std::size_t get_allocation_limit() const;
};

static_assert(cuda::mr::resource_with<limiting_resource_adaptor, cuda::mr::device_accessible>,
              "limiting_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
