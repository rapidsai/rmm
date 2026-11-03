/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cudaTypedefs.h>
#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
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
 * @brief Lock-free per-device cache of a boolean device capability.
 *
 * Device capabilities are invariant for the lifetime of the process, so each entry is only ever
 * written with the same value. Concurrent first queries may both query CUDA and store, which is
 * harmless, so relaxed atomics suffice for both reads and writes. Failed queries and out-of-range
 * device ids are not cached.
 */
class per_device_capability {
 public:
  per_device_capability()
  {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) { device_count = 0; }
    cache_ = std::vector<std::atomic<std::int8_t>>(static_cast<std::size_t>(device_count));
  }

  /**
   * @brief Returns the cached capability for `device_id`, querying it on first use.
   *
   * @param device_id The CUDA device to query
   * @param query Callable returning the capability, or `std::nullopt` if the query failed
   * @return The capability, or false if the query failed
   */
  template <typename Query>
  bool get(cuda_device_id device_id, Query&& query)
  {
    auto const index    = static_cast<std::size_t>(device_id.value());
    auto const in_range = device_id.value() >= 0 and index < cache_.size();
    if (in_range) {
      auto const cached = cache_[index].load(std::memory_order_relaxed);
      if (cached != unknown) { return cached == supported; }
    }

    std::optional<bool> const result = std::forward<Query>(query)();
    if (!result.has_value()) { return false; }
    if (in_range) {
      cache_[index].store(*result ? supported : unsupported, std::memory_order_relaxed);
    }
    return *result;
  }

 private:
  enum : std::int8_t { unknown, unsupported, supported };
  std::vector<std::atomic<std::int8_t>> cache_;
};

/**
 * @brief Returns whether a boolean device attribute equals 1, or `std::nullopt` if the query fails.
 *
 * @param attribute The device attribute to query
 * @param device_id The CUDA device to query
 * @return Whether the attribute equals 1, or `std::nullopt` on failure
 */
inline std::optional<bool> query_device_flag(cudaDeviceAttr attribute, cuda_device_id device_id)
{
  int value{};
  if (cudaDeviceGetAttribute(&value, attribute, device_id.value()) != cudaSuccess) {
    return std::nullopt;
  }
  return value == 1;
}

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
    static per_device_capability cache;
    return cache.get(device_id, [device_id] {
      return query_device_flag(cudaDevAttrMemoryPoolsSupported, device_id);
    });
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
 * pool property on the present CUDA driver version and device hardware.
 *
 * @note This function returns `false` if the version of cudart that RMM was compiled with is too
 * low (see `RMM_MIN_HWDECOMPRESS_CUDA_VERSION`).
 */
// This suppression was needed due to a false positive warning from nvcc. We
// should be able to remove it altogether once we rework the thrust allocator.
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20011
#endif
struct hwdecompress {
  /**
   * @brief Check hardware decompression support on the specified device.
   *
   * @param device_id The CUDA device to query
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_supported(cuda_device_id device_id)
  {
#if CUDART_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
    // Check if hardware decompression is supported (requires CUDA 12.8 driver or higher)
    static bool driver_supported = []() {
      int driver_version{};
      RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
      return driver_version >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION;
    }();
    if (!driver_supported) { return false; }

    // The runtime API has no enumerator for the decompression device attribute.
    static auto const get_attribute = []() -> PFN_cuDeviceGetAttribute_v2000 {
      void* function{};
      cudaDriverEntryPointQueryResult status{};
      auto const result = cudaGetDriverEntryPointByVersion("cuDeviceGetAttribute",
                                                           &function,
                                                           RMM_MIN_HWDECOMPRESS_CUDA_VERSION,
                                                           cudaEnableDefault,
                                                           &status);
      if (result != cudaSuccess || status != cudaDriverEntryPointSuccess) { return nullptr; }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      return reinterpret_cast<PFN_cuDeviceGetAttribute_v2000>(function);
    }();
    if (get_attribute == nullptr) { return false; }

    static per_device_capability cache;
    return cache.get(device_id, [device_id]() -> std::optional<bool> {
      int algorithm_mask{};
      auto const result = get_attribute(
        &algorithm_mask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device_id.value());
      if (result != CUDA_SUCCESS) { return std::nullopt; }
      return algorithm_mask != CU_MEM_DECOMPRESS_UNSUPPORTED;
    });
#else
    (void)device_id;
    return false;
#endif
  }

  /**
   * @brief Check hardware decompression support on the current device.
   *
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_supported() { return is_supported(rmm::get_current_cuda_device()); }
};
#ifdef __CUDACC__
#pragma nv_diagnostic pop
#endif

/**
 * @brief Check if a device supports concurrent managed access.
 * Concurrent managed access is required for prefetching to work.
 */
struct concurrent_managed_access {
  /**
   * @brief Check concurrent managed access support on the specified device.
   *
   * @param device_id The CUDA device to query
   * @return true if the device supports concurrent managed access, false otherwise
   */
  static bool is_supported(cuda_device_id device_id)
  {
    static per_device_capability cache;
    return cache.get(device_id, [device_id] {
      return query_device_flag(cudaDevAttrConcurrentManagedAccess, device_id);
    });
  }

  /**
   * @brief Check concurrent managed access support on the current device.
   *
   * @return true if the device supports concurrent managed access, false otherwise
   */
  static bool is_supported() { return is_supported(rmm::get_current_cuda_device()); }
};

/**
 * @brief Determine at runtime if the CUDA driver/runtime supports the stream-ordered
 * managed memory allocator functions.
 *
 * Stream-ordered managed memory pools were introduced in CUDA 13.0.
 */
struct runtime_async_managed_alloc {
  /**
   * @brief Check stream-ordered managed memory pool support on the specified device.
   *
   * @param device_id The CUDA device to query
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_supported(cuda_device_id device_id)
  {
    static auto const versions_supported{[] {
      // CUDA 13.0 or higher is required for async managed memory pools
      int cuda_driver_version{};
      auto driver_result = cudaDriverGetVersion(&cuda_driver_version);
      int cuda_runtime_version{};
      auto runtime_result = cudaRuntimeGetVersion(&cuda_runtime_version);
      return driver_result == cudaSuccess and runtime_result == cudaSuccess and
             cuda_driver_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION and
             cuda_runtime_version >= RMM_MIN_ASYNC_MANAGED_ALLOC_CUDA_VERSION;
    }()};
    // Concurrent managed access is required for async managed memory pools
    return versions_supported and concurrent_managed_access::is_supported(device_id);
  }

  /**
   * @brief Check stream-ordered managed memory pool support on the current device.
   *
   * @return true if supported
   * @return false if unsupported
   */
  static bool is_supported() { return is_supported(rmm::get_current_cuda_device()); }
};

/**
 * @brief Check if a device is an integrated memory system.
 */
struct device_integrated_memory {
  /**
   * @brief Check whether the specified device is an integrated memory system.
   *
   * @param device_id The CUDA device to query
   * @return true if the device is an integrated memory system, false otherwise
   */
  static bool is_supported(cuda_device_id device_id)
  {
    static per_device_capability cache;
    return cache.get(device_id,
                     [device_id] { return query_device_flag(cudaDevAttrIntegrated, device_id); });
  }

  /**
   * @brief Check whether the current device is an integrated memory system.
   *
   * @return true if the device is an integrated memory system, false otherwise
   */
  static bool is_supported() { return is_supported(rmm::get_current_cuda_device()); }
};

}  // namespace detail
RMM_NAMESPACE_END
