/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/mr/system_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <fstream>
#include <string>

namespace rmm::test {

/**
 * @brief Returns true if running under Windows Subsystem for Linux (WSL).
 */
inline bool is_wsl()
{
  std::ifstream proc_version("/proc/version");
  if (proc_version.is_open()) {
    std::string line;
    std::getline(proc_version, line);
    return line.find("microsoft") != std::string::npos ||
           line.find("Microsoft") != std::string::npos;
  }
  return false;
}

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
