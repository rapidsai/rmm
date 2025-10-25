/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/mr/device/system_memory_resource.hpp>

#include <cuda_runtime_api.h>

namespace rmm::test {

/**
 * @brief Returns if a pointer points to a device memory or managed memory
 * allocation.
 */
inline bool is_device_accessible_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return attributes.devicePointer != nullptr;
}

inline bool is_host_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return attributes.hostPointer != nullptr || attributes.type == cudaMemoryTypeUnregistered;
}

inline bool is_properly_aligned(void* ptr)
{
  if (is_host_memory(ptr)) { return rmm::is_pointer_aligned(ptr, rmm::RMM_DEFAULT_HOST_ALIGNMENT); }
  return rmm::is_pointer_aligned(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

}  // namespace rmm::test
