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

#include <rmm/aligned.hpp>

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
  return (attributes.type == cudaMemoryTypeDevice) or (attributes.type == cudaMemoryTypeManaged) or
         ((attributes.type == cudaMemoryTypeHost) and (attributes.devicePointer != nullptr));
}

inline bool is_host_memory(void* ptr)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, ptr)) { return false; }
  return attributes.type == cudaMemoryTypeHost;
}

inline bool is_properly_aligned(void* ptr)
{
  if (is_host_memory(ptr)) { return rmm::is_pointer_aligned(ptr, rmm::RMM_DEFAULT_HOST_ALIGNMENT); }
  return rmm::is_pointer_aligned(ptr, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

}  // namespace rmm::test
