/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/format.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {

namespace detail {
/**
 * @brief Check if system allocated memory (SAM) is supported on the specified device.
 *
 * @param device_id The device to check
 * @return true if SAM is supported on the device, false otherwise
 */
static bool is_system_memory_supported(cuda_device_id device_id)
{
  // Check if pageable memory access is supported
  int pageableMemoryAccess;
  RMM_CUDA_TRY(cudaDeviceGetAttribute(
    &pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, device_id.value()));
  return pageableMemoryAccess == 1;
}
}  // namespace detail

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */
/**
 * @brief Memory resource that uses malloc/free for allocation/deallocation.
 *
 * There are two flavors of hardware/software environments that support accessing system allocated
 * memory (SAM) from the GPU: HMM and ATS.
 *
 * Heterogeneous Memory Management (HMM) is a software-based solution for PCIe-connected GPUs on
 * x86 systems. Requirements:
 *   - NVIDIA CUDA 12.2 with the open-source r535_00 driver or newer.
 *   - A sufficiently recent Linux kernel: 6.1.24+, 6.2.11+, or 6.3+.
 *   - A GPU with one of the following supported architectures: NVIDIA Turing, NVIDIA Ampere,
 *     NVIDIA Ada Lovelace, NVIDIA Hopper, or newer.
 *   - A 64-bit x86 CPU.
 *
 *  For more information, see
 *  https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/.
 *
 *  Address Translation Services (ATS) is a hardware/software solution for the Grace Hopper
 *  Superchip that uses the NVLink Chip-2-Chip (C2C) interconnect to provide coherent memory. For
 *  more information, see
 *  https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/.
 */
class system_memory_resource final {
 public:
  system_memory_resource()
  {
    RMM_EXPECTS(rmm::mr::detail::is_system_memory_supported(rmm::get_current_cuda_device()),
                "System memory allocator is not supported with this hardware/software version.");
  }
  ~system_memory_resource()                             = default;
  system_memory_resource(system_memory_resource const&) = default;  ///< @default_copy_constructor
  system_memory_resource(system_memory_resource&&)      = default;  ///< @default_copy_constructor
  system_memory_resource& operator=(system_memory_resource const&) =
    default;  ///< @default_copy_assignment{system_memory_resource}
  system_memory_resource& operator=(system_memory_resource&&) =
    default;  ///< @default_move_assignment{system_memory_resource}

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
    try {
      return rmm::detail::aligned_host_allocate(
        bytes, CUDA_ALLOCATION_ALIGNMENT, [](std::size_t size) { return ::operator new(size); });
    } catch (std::bad_alloc const& e) {
      auto const msg = std::string("Failed to allocate ") + rmm::detail::format_bytes(bytes) +
                       std::string("of memory: ") + e.what();
      RMM_FAIL(msg.c_str(), rmm::out_of_memory);
    }
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   *
   * This function synchronizes the stream before deallocating the memory.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   *              that was passed to the `allocate` call that returned `ptr`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `ptr`
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    // With `cudaFree`, the CUDA runtime keeps track of dependent operations and does implicit
    // synchronization. However, with SAM, since `free` is immediate, we need to wait for in-flight
    // CUDA operations to finish before freeing the memory, to avoid potential use-after-free errors
    // or race conditions.
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));

    rmm::detail::aligned_host_deallocate(
      ptr, bytes, CUDA_ALLOCATION_ALIGNMENT, [](void* ptr) { ::operator delete(ptr); });
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
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
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
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `system_memory_resource` provides device-accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `system_memory_resource` provides host-accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(system_memory_resource const&,
                                         cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Compare this resource to another.
   *
   * All instances of system_memory_resource are equivalent.
   *
   * @return true Always
   */
  [[nodiscard]] bool operator==(system_memory_resource const&) const noexcept { return true; }

  /**
   * @copydoc operator==
   */
  [[nodiscard]] bool operator!=(system_memory_resource const&) const noexcept { return false; }
};

// static property checks
static_assert(cuda::mr::synchronous_resource<system_memory_resource>);
static_assert(cuda::mr::resource<system_memory_resource>);
static_assert(
  cuda::mr::synchronous_resource_with<system_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<system_memory_resource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<system_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<system_memory_resource, cuda::mr::host_accessible>);
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
