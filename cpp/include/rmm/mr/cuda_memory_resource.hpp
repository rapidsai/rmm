/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/device_memory_resource.hpp>

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
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation.
 */
class cuda_memory_resource final : public device_memory_resource {
 public:
  cuda_memory_resource()                            = default;
  ~cuda_memory_resource() override                  = default;
  cuda_memory_resource(cuda_memory_resource const&) = default;  ///< @default_copy_constructor
  cuda_memory_resource(cuda_memory_resource&&)      = default;  ///< @default_move_constructor
  cuda_memory_resource& operator=(cuda_memory_resource const&) =
    default;  ///< @default_copy_assignment{cuda_memory_resource}
  cuda_memory_resource& operator=(cuda_memory_resource&&) =
    default;  ///< @default_move_assignment{cuda_memory_resource}

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
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    (void)stream;
    (void)alignment;
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&ptr, bytes), bytes);
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
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    (void)stream;
    (void)bytes;
    (void)alignment;
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
    return allocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, bytes, alignment);
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
   * This property declares that a `cuda_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(cuda_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  // -- Legacy device_memory_resource overrides (delegates to CCCL interface) --
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return allocate(stream, bytes);
  }

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override
  {
    deallocate(stream, ptr, bytes);
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<cuda_memory_resource const*>(&other) != nullptr;
  }
};

// static property checks
static_assert(cuda::mr::synchronous_resource<cuda_memory_resource>);
static_assert(cuda::mr::resource<cuda_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<cuda_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_memory_resource, cuda::mr::device_accessible>);

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
