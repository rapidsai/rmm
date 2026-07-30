/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */
/**
 * @brief Memory resource that uses cudaMalloc/Free for allocation/deallocation.
 */
class RMM_EXPORT cuda_memory_resource final {
 public:
  cuda_memory_resource()                            = default;
  ~cuda_memory_resource()                           = default;
  cuda_memory_resource(cuda_memory_resource const&) = default;  ///< @default_copy_constructor
  cuda_memory_resource(cuda_memory_resource&&)      = default;  ///< @default_move_constructor
  cuda_memory_resource& operator=(cuda_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_memory_resource}
  cuda_memory_resource& operator=(cuda_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_memory_resource}

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * The stream argument is ignored.
   *
   * @param stream This argument is ignored
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate([[maybe_unused]] cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   *
   * The stream argument is ignored.
   *
   * @param stream This argument is ignored
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Allocates memory of size at least \p bytes synchronously.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocate memory pointed to by \p ptr synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `cuda_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Compare this resource to another.
   *
   * All instances of cuda_memory_resource are equivalent.
   *
   * @return true Always
   */
  [[nodiscard]] bool operator==(cuda_memory_resource const&) const noexcept;

  /**
   * @copydoc operator==
   */
  [[nodiscard]] bool operator!=(cuda_memory_resource const&) const noexcept;
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_memory_resource>);
static_assert(cuda::mr::resource<cuda_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<cuda_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_memory_resource, cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
