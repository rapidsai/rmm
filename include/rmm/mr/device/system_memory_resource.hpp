/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm::mr {

namespace detail {
/**
 * @brief Check if system allocated memory (SAM) is supported on the specified device.
 *
 * @param device_id The device to check
 * @return true if SAM is supported on the device, false otherwise
 */
static bool is_system_memory_supported(cuda_device_id device_id)
{
  int pageableMemoryAccess;
  RMM_CUDA_TRY(cudaDeviceGetAttribute(
    &pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, device_id.value()));
  return pageableMemoryAccess == 1;
}
}  // namespace detail

/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */
/**
 * @brief `device_memory_resource` derived class that uses malloc/free for allocation/deallocation.
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
class system_memory_resource final : public device_memory_resource {
 public:
  system_memory_resource()
  {
    RMM_EXPECTS(rmm::mr::detail::is_system_memory_supported(rmm::get_current_cuda_device()),
                "System memory allocator is not supported with this hardware/software version.");
  }
  ~system_memory_resource() override                    = default;
  system_memory_resource(system_memory_resource const&) = default;  ///< @default_copy_constructor
  system_memory_resource(system_memory_resource&&)      = default;  ///< @default_copy_constructor
  system_memory_resource& operator=(system_memory_resource const&) =
    default;  ///< @default_copy_assignment{system_memory_resource}
  system_memory_resource& operator=(system_memory_resource&&) =
    default;  ///< @default_move_assignment{system_memory_resource}

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * The stream argument is ignored.
   *
   * @param bytes The size of the allocation
   * @param stream This argument is ignored
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, [[maybe_unused]] cuda_stream_view stream) override
  {
    try {
      return rmm::detail::aligned_host_allocate(
        bytes, CUDA_ALLOCATION_ALIGNMENT, [](std::size_t size) { return ::operator new(size); });
    } catch (std::bad_alloc const& e) {
      RMM_FAIL("Failed to allocate memory: " + std::string{e.what()}, rmm::out_of_memory);
    }
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * The stream argument is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   *              that was passed to the `allocate` call that returned `ptr`.
   * @param stream This argument is ignored
   */
  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] cuda_stream_view stream) override
  {
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, CUDA_ALLOCATION_ALIGNMENT, [](void* ptr) { ::operator delete(ptr); });
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two system_memory_resources always compare equal, because they can each deallocate memory
   * allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<system_memory_resource const*>(&other) != nullptr;
  }
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `system_memory_resource` provides device-accessible memory
   */
  friend void get_property(system_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `system_memory_resource` provides host-accessible memory
   */
  friend void get_property(system_memory_resource const&, cuda::mr::host_accessible) noexcept {}
};

// static property checks
static_assert(cuda::mr::async_resource_with<system_memory_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::async_resource_with<system_memory_resource, cuda::mr::host_accessible>);
/** @} */  // end of group
}  // namespace rmm::mr
