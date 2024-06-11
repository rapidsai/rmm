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

#include <cstddef>
#include <limits>
#include <optional>

namespace rmm::mr {

namespace detail {
struct sam {
  static bool is_supported()
  {
    int pageableMemoryAccess;
    RMM_CUDA_TRY(cudaDeviceGetAttribute(&pageableMemoryAccess,
                                        cudaDevAttrPageableMemoryAccess,
                                        rmm::get_current_cuda_device().value()));
    return pageableMemoryAccess == 1;
  }
};
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
 * memory from the GPU: HMM and ATS.
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
  /**
   * @brief Construct a new system memory resource.
   *
   * By default if no parameters are specified, this memory resource pass through to malloc/free
   * for allocation/deallocation.
   *
   * However, when GPU memory is over-subscribed, system allocated memory will migrate to the GPU
   * and cannot migrate back, thus causing other CUDA calls to fail with out-of-memory errors. To
   * work around this problem, we can reserve some GPU memory as headroom for other CUDA calls.
   * Doing this check can be expensive, so only large buffer above the given threshold will be
   * checked.
   *
   * @param headroom_size Size of the reserved GPU memory as headroom
   * @param threshold_size Size of the allocation above which to check for headroom
   */
  explicit system_memory_resource(std::optional<std::size_t> headroom_size  = std::nullopt,
                                  std::optional<std::size_t> threshold_size = std::nullopt)
  {
    RMM_EXPECTS(rmm::mr::detail::sam::is_supported(),
                "System memory allocator is not supported with this hardware/software version.");
    headroom_size_ = headroom_size.value_or(0);
    if (headroom_size_ == 0) {
      // If headroom is not specified, always pass through to malloc.
      threshold_size_ = threshold_size.value_or(std::numeric_limits<std::size_t>::max());
    } else {
      // If headroom is specified, by default we check every allocation.
      threshold_size_ = threshold_size.value_or(0);
    }
  }

  ~system_memory_resource() override                    = default;
  system_memory_resource(system_memory_resource const&) = default;  ///< @default_copy_constructor
  system_memory_resource(system_memory_resource&&)      = default;  ///< @default_move_constructor
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
    auto const aligned{rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT)};
    void* ptr{malloc(aligned)};

    if (bytes >= threshold_size_) {
      auto const free        = rmm::available_device_memory().first;
      auto const allocatable = std::max(free - headroom_size_, 0UL);
      auto const gpu_portion = std::min(allocatable, aligned);
      auto const cpu_portion = aligned - gpu_portion;
      if (gpu_portion != 0) {
        RMM_CUDA_TRY(cudaMemAdvise(ptr,
                                   gpu_portion,
                                   cudaMemAdviseSetPreferredLocation,
                                   rmm::get_current_cuda_device().value()));
      }
      if (cpu_portion != 0) {
        RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(ptr) + gpu_portion,
                                   cpu_portion,
                                   cudaMemAdviseSetPreferredLocation,
                                   cudaCpuDeviceId));
      }
    }
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * The stream argument is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes This argument is ignored
   * @param stream This argument is ignored
   */
  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] cuda_stream_view stream) override
  {
    free(ptr);
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two cuda_memory_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<system_memory_resource const*>(&other);
    if (cast == nullptr) { return false; }
    return headroom_size_ == cast->headroom_size_ && threshold_size_ == cast->threshold_size_;
  }

  /// Size of GPU memory reserved as headroom.
  std::size_t headroom_size_;
  /// Size of allocation above which to check for headroom.
  std::size_t threshold_size_;
};
/** @} */  // end of group
}  // namespace rmm::mr
