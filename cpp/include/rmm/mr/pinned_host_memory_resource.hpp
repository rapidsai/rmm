/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cccl_adaptors.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/nvtx/ranges.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Memory resource class for allocating pinned host memory.
 *
 * This class uses CUDA's `cudaHostAlloc` to allocate pinned host memory. It implements the
 * `cuda::mr::memory_resource` and `cuda::mr::device_memory_resource` concepts, and
 * the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.
 */
class pinned_host_memory_resource final : public device_memory_resource {
 public:
  pinned_host_memory_resource()           = default;
  ~pinned_host_memory_resource() override = default;
  pinned_host_memory_resource(pinned_host_memory_resource const&) =
    default;  ///< @default_copy_constructor
  pinned_host_memory_resource(pinned_host_memory_resource&&) =
    default;  ///< @default_move_constructor
  pinned_host_memory_resource& operator=(pinned_host_memory_resource const&) =
    default;  ///< @default_copy_assignment{pinned_host_memory_resource}
  pinned_host_memory_resource& operator=(pinned_host_memory_resource&&) =
    default;  ///< @default_move_assignment{pinned_host_memory_resource}

 private:
  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled due to to a
   * CUDA out of memory error.
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled due to any other
   * reason.
   *
   * The stream argument is ignored.
   *
   * @param bytes The size, in bytes, of the allocation.
   * @param stream CUDA stream on which to perform the allocation (ignored).
   *
   * @return Pointer to the newly allocated memory.
   */
  void* do_allocate(std::size_t bytes, [[maybe_unused]] cuda_stream_view stream) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    // TODO: Use the alignment parameter as an argument to do_allocate
    std::size_t constexpr alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    return rmm::detail::aligned_host_allocate(bytes, alignment, [](std::size_t size) {
      void* ptr{nullptr};
      RMM_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, size, cudaHostAllocDefault), size);
      return ptr;
    });
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * The stream argument is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream This argument is ignored.
   */
  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     [[maybe_unused]] cuda_stream_view stream) noexcept override
  {
    // TODO: Use the alignment parameter as an argument to do_deallocate
    std::size_t constexpr alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr)); });
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two pinned_host_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<pinned_host_memory_resource const*>(&other) != nullptr;
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  friend void get_property(pinned_host_memory_resource const&, cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  friend void get_property(pinned_host_memory_resource const&, cuda::mr::host_accessible) noexcept
  {
  }
};

static_assert(rmm::detail::polyfill::async_resource_with<pinned_host_memory_resource,
                                                         cuda::mr::device_accessible,
                                                         cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
