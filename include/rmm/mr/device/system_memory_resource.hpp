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
 */
class system_memory_resource final : public device_memory_resource {
 public:
  explicit system_memory_resource(std::optional<std::size_t> headroom_size  = std::nullopt,
                                  std::optional<std::size_t> threshold_size = std::nullopt)
  {
    headroom_size_  = rmm::align_down(headroom_size.value_or(0), rmm::CUDA_ALLOCATION_ALIGNMENT);
    threshold_size_ = threshold_size.value_or(std::numeric_limits<std::size_t>::max());
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
      auto const [free, _]   = rmm::available_device_memory();
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
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream This argument is ignored.
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
