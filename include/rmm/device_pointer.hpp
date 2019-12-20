/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/rmm.hpp>

#include <cuda_runtime_api.h>

namespace rmm {
/**----------------------------------------------------------------------------*
 * @file device_pointer.hpp
 * @brief RAII wrapper for an RMM allocated raw device memory pointer
 *
 * This class wraps a raw device memory pointer and frees it when the instance
 * of the class goes out of scope. Effectively, `device_pointer` takes
 * ownership of the memory pointed to by the pointer.
 *
 * Examples:
 * ```
 * TBD
 * ```
 *---------------------------------------------------------------------------**/
class device_pointer {
 public:
  /**--------------------------------------------------------------------------*
   * @brief Constructs a new device_pointer from a given pointer
   *
   * @param ptr Pointer to the device memory.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   *-------------------------------------------------------------------------**/
  device_pointer(void const* ptr, cudaStream_t stream = 0)
      : _ptr{ptr}, _stream{stream} {}

  /**--------------------------------------------------------------------------*
   * @brief Destroy the `device_pointer` object and free the underlying memory.
   *-------------------------------------------------------------------------**/
  ~device_pointer() noexcept {
    free(_data, _stream)
    _data = nullptr;
    _stream = 0;
  }

  /**--------------------------------------------------------------------------*
   * @brief Returns pointer being managed
   *-------------------------------------------------------------------------**/
  void const* ptr() const noexcept { return _ptr; }

  /**--------------------------------------------------------------------------*
   * @brief Returns pointer being managed
   *-------------------------------------------------------------------------**/
  void* ptr() noexcept { return _ptr; }

  /**--------------------------------------------------------------------------*
   * @brief Returns stream of pointer being managed
   *-------------------------------------------------------------------------**/
  cudaStream_t stream() const noexcept { return _stream; }

 private:
  void* _ptr{nullptr};     ///< Pointer being managed
  cudaStream_t _stream{};   ///< Stream to use for device memory deallocation
};
}  // namespace rmm
