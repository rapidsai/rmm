/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */
/**
 * @brief Memory resource that uses cudaMallocManaged/Free for allocation/deallocation.
 */
class managed_memory_resource final {
 public:
  managed_memory_resource()                               = default;
  ~managed_memory_resource()                              = default;
  managed_memory_resource(managed_memory_resource const&) = default;  ///< @default_copy_constructor
  managed_memory_resource(managed_memory_resource&&)      = default;  ///< @default_move_constructor
  managed_memory_resource& operator=(managed_memory_resource const&) =
    default;  ///< @default_copy_assignment{managed_memory_resource}
  managed_memory_resource& operator=(managed_memory_resource&&) =
    default;  ///< @default_move_assignment{managed_memory_resource}

  // -- CCCL memory resource interface (hides device_memory_resource versions) --

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
  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    // FIXME: Unlike cudaMalloc, cudaMallocManaged will throw an error for 0
    // size allocations.
    if (bytes == 0) { return nullptr; }

    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMallocManaged(&ptr, bytes), bytes);
    return ptr;
  }

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
                  [[maybe_unused]] std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr));
  }

  /**
   * @brief Allocates memory of size at least \p bytes synchronously.
   *
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = allocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(0)));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, ptr, bytes, alignment);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `managed_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `managed_memory_resource` provides host accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(managed_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Compare this resource to another.
   *
   * All instances of managed_memory_resource are equivalent.
   *
   * @return true Always
   */
  [[nodiscard]] bool operator==(managed_memory_resource const&) const noexcept { return true; }

  /**
   * @copydoc operator==
   */
  [[nodiscard]] bool operator!=(managed_memory_resource const&) const noexcept { return false; }
};

// static property checks
static_assert(cuda::mr::synchronous_resource<managed_memory_resource>);
static_assert(cuda::mr::resource<managed_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<managed_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<managed_memory_resource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<managed_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<managed_memory_resource, cuda::mr::host_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
