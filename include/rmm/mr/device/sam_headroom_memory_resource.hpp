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
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/system_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */
/**
 * @brief Resource that uses system memory resource to allocate memory with a headroom.
 *
 * System allocated memory (SAM) can be migrated to the GPU, but is never migrated back the host. If
 * GPU memory is over-subscribed, this can cause other CUDA calls to fail with out-of-memory errors.
 * To work around this problem, when using a system memory resource, we reserve some GPU memory as
 * headroom for other CUDA calls, and only conditionally set its preferred location to the GPU if
 * the allocation would not eat into the headroom.
 *
 * Since doing this check on every allocation can be expensive, the caller may choose to use other
 * allocators (e.g. `binning_memory_resource`) for small allocations, and use this allocator for
 * large allocations only.
 */
class sam_headroom_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Construct a headroom memory resource.
   *
   * @param headroom Size of the reserved GPU memory as headroom
   */
  explicit sam_headroom_memory_resource(std::size_t headroom) : system_mr_{}, headroom_{headroom} {}

  sam_headroom_memory_resource()                                               = delete;
  ~sam_headroom_memory_resource() override                                     = default;
  sam_headroom_memory_resource(sam_headroom_memory_resource const&)            = delete;
  sam_headroom_memory_resource(sam_headroom_memory_resource&&)                 = delete;
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource const&) = delete;
  sam_headroom_memory_resource& operator=(sam_headroom_memory_resource&&)      = delete;

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
    void* pointer = system_mr_.allocate_async(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, stream);

    auto const free        = rmm::available_device_memory().first;
    auto const allocatable = free > headroom_ ? free - headroom_ : 0UL;
    auto const gpu_portion =
      rmm::align_down(std::min(allocatable, bytes), rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto const cpu_portion = bytes - gpu_portion;
    if (gpu_portion != 0) {
      RMM_CUDA_TRY(cudaMemAdvise(pointer,
                                 gpu_portion,
                                 cudaMemAdviseSetPreferredLocation,
                                 rmm::get_current_cuda_device().value()));
    }
    if (cpu_portion != 0) {
      RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(pointer) + gpu_portion,
                                 cpu_portion,
                                 cudaMemAdviseSetPreferredLocation,
                                 cudaCpuDeviceId));
    }

    return pointer;
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
    system_mr_.deallocate_async(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT, stream);
  }

  /**
   * @brief Compare this resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<sam_headroom_memory_resource const*>(&other);
    if (cast == nullptr) { return false; }
    return headroom_ == cast->headroom_;
  }

  ///< The system memory resource used for satisfying allocation requests
  system_memory_resource system_mr_;
  ///< Size of GPU memory reserved as headroom
  std::size_t headroom_;
};
/** @} */  // end of group
}  // namespace rmm::mr
