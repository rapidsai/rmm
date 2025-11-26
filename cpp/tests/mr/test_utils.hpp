/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/mr/system_memory_resource.hpp>

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

inline bool is_properly_aligned(void* ptr)
{
  return rmm::is_pointer_aligned(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

}  // namespace rmm::test
