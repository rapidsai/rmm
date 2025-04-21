/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <rmm/detail/export.hpp>

#include <cuda_runtime_api.h>

#include <dlfcn.h>

namespace RMM_NAMESPACE {
namespace detail {

/**
 * @brief Minimum CUDA driver version for hardware decompression support
 */
#define RMM_MIN_HWDECOMPRESS_CUDA_DRIVER_VERSION 12080

/**
 * @brief Determine at runtime if the CUDA driver supports the stream-ordered
 * memory allocator functions.
 *
 * This allows RMM users to compile/link against CUDA 11.2+ and run with
 * older drivers.
 */

struct runtime_async_alloc {
  static bool is_supported()
  {
    static auto driver_supports_pool{[] {
      int cuda_pool_supported{};
      auto result = cudaDeviceGetAttribute(&cuda_pool_supported,
                                           cudaDevAttrMemoryPoolsSupported,
                                           rmm::get_current_cuda_device().value());
      return result == cudaSuccess and cuda_pool_supported == 1;
    }()};
    return driver_supports_pool;
  }

  /**
   * @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
   * CUDA driver/runtime version.
   *
   * @param handle_type An IPC export handle type to check for support.
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_export_handle_type_supported(cudaMemAllocationHandleType handle_type)
  {
    int supported_handle_types_bitmask{};
    if (cudaMemHandleTypeNone != handle_type) {
      auto const result = cudaDeviceGetAttribute(&supported_handle_types_bitmask,
                                                 cudaDevAttrMemoryPoolSupportedHandleTypes,
                                                 rmm::get_current_cuda_device().value());

      // Don't throw on cudaErrorInvalidValue
      auto const unsupported_runtime = (result == cudaErrorInvalidValue);
      if (unsupported_runtime) return false;
      // throw any other error that may have occurred
      RMM_CUDA_TRY(result);
    }
    return (supported_handle_types_bitmask & handle_type) == handle_type;
  }

  /**
   * @brief Check whether `cudaMemPoolCreateUsageHwDecompress` is a supported
   * pool property on the present CUDA driver version.
   *
   * Requires RMM to be built with a supported CUDA version 12.8+, otherwise
   * this always returns false.
   *
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_hwdecompress_supported()
  {
#if defined(CUDA_VERSION) && CUDA_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_DRIVER_VERSION
    // Check if hardware decompression is supported (requires CUDA 12.8 driver or higher)
    static bool is_supported = []() {
      int driver_version{};
      RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
      return driver_version >= RMM_MIN_HWDECOMPRESS_CUDA_DRIVER_VERSION;
    }();
    return is_supported;
#else
    return false;
#endif
  }
};

}  // namespace detail
}  // namespace RMM_NAMESPACE
