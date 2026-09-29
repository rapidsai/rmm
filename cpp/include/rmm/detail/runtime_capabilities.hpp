/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

RMM_NAMESPACE_BEGIN
namespace detail {

/**
 * @brief Minimum CUDA version for hardware decompression support.
 *
 * The driver version used at runtime must be at least this value. Moreover, the
 * compile-time cudart version must also be at least this value.
 */
#define RMM_MIN_HWDECOMPRESS_CUDA_VERSION 12080

/**
 * @brief Minimum CUDA driver version for stream-ordered managed memory allocator support
 */
#define RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION 13000

/**
 * @brief Determine at runtime if the CUDA driver supports the stream-ordered
 * memory allocator functions.
 *
 * Stream-ordered memory pools were introduced in CUDA 11.2. This allows RMM
 * users to compile/link against newer CUDA versions and run with older
 * drivers.
 */
struct runtime_async_alloc {
  /**
   * @brief Determine whether the specified device supports stream-ordered memory pools.
   *
   * Successful queries are cached per device. Reads and writes of the cache are lock-free.
   *
   * @param device_id The CUDA device to query
   * @return true if supported
   * @return false if unsupported or if the attribute query fails
   */
  static bool is_supported(cuda_device_id device_id)
  {
    enum : std::int8_t { unknown, unsupported, supported };
    static std::vector<std::atomic<std::int8_t>> cache = [] {
      int device_count{};
      if (cudaGetDeviceCount(&device_count) != cudaSuccess) { device_count = 0; }
      return std::vector<std::atomic<std::int8_t>>(static_cast<std::size_t>(device_count));
    }();

    auto const index    = static_cast<std::size_t>(device_id.value());
    auto const in_range = device_id.value() >= 0 and index < cache.size();
    // Each entry is written only with the same device-invariant value, so relaxed ordering
    // suffices and concurrent writers are harmless.
    if (in_range) {
      auto const cached = cache[index].load(std::memory_order_relaxed);
      if (cached != unknown) { return cached == supported; }
    }

    int cuda_pool_supported{};
    auto const result = cudaDeviceGetAttribute(
      &cuda_pool_supported, cudaDevAttrMemoryPoolsSupported, device_id.value());
    if (result != cudaSuccess) { return false; }

    auto const is_supported = cuda_pool_supported == 1;
    if (in_range) {
      cache[index].store(is_supported ? supported : unsupported, std::memory_order_relaxed);
    }
    return is_supported;
  }

  /**
   * @brief Determine whether the current device supports stream-ordered memory pools.
   *
   * @return true if supported
   * @return false if unsupported or if the attribute query fails
   */
  static bool is_supported() { return is_supported(rmm::get_current_cuda_device()); }
};

/**
 * @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
 * CUDA driver/runtime version.
 *
 * @param handle_type An IPC export handle type to check for support.
 * @return true if supported
 * @return false if unsupported
 */
struct export_handle_type {
  static bool is_supported(cudaMemAllocationHandleType handle_type)
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
};

/**
 * @brief Check whether `cudaMemPoolCreateUsageHwDecompress` is a supported
 * pool property on the present CUDA driver version.
 *
 * @note Even if this function returns `true`, hardware decompression will not be enabled
 * on async memory pools if the version of cudart that RMM was compiled with is too low
 * (see `RMM_MIN_HWDECOMPRESS_CUDA_VERSION`).
 *
 * @return true if supported
 * @return false if unsupported
 */
// This suppression was needed due to a false positive warning from nvcc. We
// should be able to remove it altogether once we rework the thrust allocator.
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20011
#endif
struct hwdecompress {
  static bool is_supported()
  {
    // Check if hardware decompression is supported (requires CUDA 12.8 driver or higher)
    static bool is_supported = []() {
      int driver_version{};
      RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
      return driver_version >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION;
    }();
    return is_supported;
  }
};
#ifdef __CUDACC__
#pragma nv_diagnostic pop
#endif

/**
 * @brief Check if the current device supports concurrent managed access.
 * Concurrent managed access is required for prefetching to work.
 *
 * @return true if the device supports concurrent managed access, false otherwise
 */
struct concurrent_managed_access {
  static bool is_supported()
  {
    static auto driver_supports_concurrent_managed_access{[] {
      int concurrentManagedAccess = 0;
      auto result                 = cudaDeviceGetAttribute(&concurrentManagedAccess,
                                           cudaDevAttrConcurrentManagedAccess,
                                           rmm::get_current_cuda_device().value());
      return result == cudaSuccess and concurrentManagedAccess == 1;
    }()};
    return driver_supports_concurrent_managed_access;
  }
};

/**
 * @brief Determine at runtime if the CUDA driver/runtime supports the stream-ordered
 * managed memory allocator functions.
 *
 * Stream-ordered managed memory pools were introduced in CUDA 13.0.
 */
struct runtime_async_managed_alloc {
  static bool is_supported()
  {
    static auto supports_async_managed_pool{[] {
      // Concurrent managed access is required for async managed memory pools
      if (not concurrent_managed_access::is_supported()) { return false; }
      // CUDA 13.0 or higher is required for async managed memory pools
      int cuda_driver_version{};
      auto driver_result = cudaDriverGetVersion(&cuda_driver_version);
      int cuda_runtime_version{};
      auto runtime_result = cudaRuntimeGetVersion(&cuda_runtime_version);
      return driver_result == cudaSuccess and runtime_result == cudaSuccess and
             cuda_driver_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION and
             cuda_runtime_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION;
    }()};
    return supports_async_managed_pool;
  }
};

/**
 * @brief Check if the current device is an integrated memory system.
 *
 * @return true if the device is an integrated memory system, false otherwise
 */
struct device_integrated_memory {
  static bool is_supported()
  {
    static auto is_integrated{[] {
      int integrated = 0;
      auto result    = cudaDeviceGetAttribute(
        &integrated, cudaDevAttrIntegrated, rmm::get_current_cuda_device().value());
      return result == cudaSuccess and integrated == 1;
    }()};
    return is_integrated;
  }
};

}  // namespace detail
RMM_NAMESPACE_END
