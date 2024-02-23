/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */
/**
 * @brief `device_memory_resource` derived class that uses
 * cudaMallocManaged/Free for allocation/deallocation.
 */
class managed_memory_resource final : public device_memory_resource {
 public:
  managed_memory_resource()                               = default;
  ~managed_memory_resource() override                     = default;
  managed_memory_resource(managed_memory_resource const&) = default;  ///< @default_copy_constructor
  managed_memory_resource(managed_memory_resource&&)      = default;  ///< @default_move_constructor
  managed_memory_resource& operator=(managed_memory_resource const&) =
    default;  ///< @default_copy_assignment{managed_memory_resource}
  managed_memory_resource& operator=(managed_memory_resource&&) =
    default;  ///< @default_move_assignment{managed_memory_resource}

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * The stream is ignored.
   *
   * @param bytes The size of the allocation
   * @param stream This argument is ignored
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, [[maybe_unused]] cuda_stream_view stream) override
  {
    // FIXME: Unlike cudaMalloc, cudaMallocManaged will throw an error for 0
    // size allocations.
    if (bytes == 0) { return nullptr; }

    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * The stream is ignored.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream This argument is ignored
   */
  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] cuda_stream_view stream) override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(ptr));
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two `managed_memory_resources` always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<managed_memory_resource const*>(&other) != nullptr;
  }
};

/** @} */  // end of group
}  // namespace rmm::mr
