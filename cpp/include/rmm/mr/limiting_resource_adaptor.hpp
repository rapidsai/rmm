/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/limiting_resource_adaptor_impl.hpp>
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
 * @brief Resource that uses an upstream resource to allocate memory and limits the total
 * allocations possible.
 *
 * Atomics are used to make the byte counter thread-safe, but note that `get_allocated_bytes`
 * may not include in-flight allocations.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RMM_EXPORT limiting_resource_adaptor
  : public device_memory_resource,
    private cuda::mr::shared_resource<detail::limiting_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::limiting_resource_adaptor_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Compare two adaptors for equality (shared-impl identity).
   *
   * @param other The other limiting_resource_adaptor to compare against.
   * @return true if both adaptors share the same underlying state.
   */
  [[nodiscard]] bool operator==(limiting_resource_adaptor const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Compare two adaptors for inequality.
   *
   * @param other The other limiting_resource_adaptor to compare against.
   * @return true if the adaptors do not share the same underlying state.
   */
  [[nodiscard]] bool operator!=(limiting_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer

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
  limiting_resource_adaptor(device_async_resource_ref upstream,
                            std::size_t allocation_limit,
                            std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Construct a new limiting resource adaptor using `upstream` to satisfy
   * allocation requests and limiting the total allocation amount possible.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param allocation_limit Maximum memory allowed for this allocator
   * @param alignment Alignment in bytes for the start of each allocated buffer
   */
  limiting_resource_adaptor(device_memory_resource* upstream,
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

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

static_assert(cuda::mr::resource_with<limiting_resource_adaptor, cuda::mr::device_accessible>,
              "limiting_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
