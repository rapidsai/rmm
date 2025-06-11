/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <rmm/detail/nvtx/ranges.hpp>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtMem.h>

namespace rmm {

struct librmm_memory_domain {
  static constexpr char const* name{"librmm_memory"};  ///< Name of the librmm domain
};

/**
 * @brief Get the nvtx domain object
 *
 * @return nvtx3::domain const&
 */
inline nvtx3::domain const& nvtx_domain() { return nvtx3::domain::get<librmm_memory_domain>(); }

/**
 * @brief Create a new nvtx heap for the allocated memory
 *
 * @param ptr Pointer to the allocated memory
 * @param size Size of the allocated memory
 * @return nvtxMemHeapHandle_t Handle to the nvtx heap
 */
inline nvtxMemHeapHandle_t create_nvtx_heap(void* ptr, std::size_t size)
{
  nvtxMemVirtualRangeDesc_t nvtxRangeDesc = {};
  nvtxRangeDesc.size                      = size;
  nvtxRangeDesc.ptr                       = ptr;

  nvtxMemHeapDesc_t nvtxHeapDesc    = {};
  nvtxHeapDesc.extCompatID          = NVTX_EXT_COMPATID_MEM;
  nvtxHeapDesc.structSize           = sizeof(nvtxMemHeapDesc_t);
  nvtxHeapDesc.usage                = NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
  nvtxHeapDesc.type                 = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
  nvtxHeapDesc.typeSpecificDescSize = sizeof(nvtxMemVirtualRangeDesc_t);
  nvtxHeapDesc.typeSpecificDesc     = &nvtxRangeDesc;

  return nvtxMemHeapRegister(nvtx_domain(), &nvtxHeapDesc);
}

/**
 * @brief Destroy the nvtx heap
 *
 * @param handle Handle to the nvtx heap
 */
inline void destroy_nvtx_heap(nvtxMemHeapHandle_t handle)
{
  nvtxMemHeapUnregister(nvtx_domain(), handle);
}

/**
 * @brief Register the memory region with the nvtx heap
 *
 * @param handle Handle to the nvtx heap
 * @param ptr Pointer to the memory region
 * @param size Size of the memory region
 */
inline void register_mem_region(nvtxMemHeapHandle_t handle, void const* ptr, std::size_t size)
{
  nvtxMemVirtualRangeDesc_t nvtxRangeDesc = {};
  nvtxRangeDesc.size                      = size;
  nvtxRangeDesc.ptr                       = ptr;

  nvtxMemRegionsRegisterBatch_t nvtxRegionsDesc = {};
  nvtxRegionsDesc.extCompatID                   = NVTX_EXT_COMPATID_MEM;
  nvtxRegionsDesc.structSize                    = sizeof(nvtxMemRegionsRegisterBatch_t);
  nvtxRegionsDesc.regionType                    = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
  nvtxRegionsDesc.heap                          = handle;
  nvtxRegionsDesc.regionCount                   = 1;
  nvtxRegionsDesc.regionDescElementSize         = sizeof(nvtxMemVirtualRangeDesc_t);
  nvtxRegionsDesc.regionDescElements            = &nvtxRangeDesc;

  nvtxMemRegionsRegister(nvtx_domain(), &nvtxRegionsDesc);
}

/**
 * @brief Unregister the memory region from the nvtx heap
 *
 * @param ptr Pointer to the memory region
 */
inline void unregister_mem_region(void const* ptr)
{
  nvtxMemRegionRef_t nvtxRegionRef;
  nvtxRegionRef.pointer = ptr;

  nvtxMemRegionsUnregisterBatch_t nvtxRegionsDesc = {};
  nvtxRegionsDesc.extCompatID                     = NVTX_EXT_COMPATID_MEM;
  nvtxRegionsDesc.structSize                      = sizeof(nvtxMemRegionsUnregisterBatch_t);
  nvtxRegionsDesc.refType                         = NVTX_MEM_REGION_REF_TYPE_POINTER;
  nvtxRegionsDesc.refCount                        = 1;
  nvtxRegionsDesc.refElementSize                  = sizeof(nvtxMemRegionRef_t);
  nvtxRegionsDesc.refElements                     = &nvtxRegionRef;

  nvtxMemRegionsUnregister(nvtx_domain(), &nvtxRegionsDesc);
}

}  // namespace rmm
