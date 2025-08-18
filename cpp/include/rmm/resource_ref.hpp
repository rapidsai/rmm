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

#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

namespace RMM_NAMESPACE {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
class device_resource_ref : public cuda::mr::resource_ref<cuda::mr::device_accessible> {
 public:
  using base_type = cuda::mr::resource_ref<cuda::mr::device_accessible>;

  /**
   * @brief Allocates memory of size at least \p bytes asynchronously.
   *
   * @param bytes The size of the allocation.
   * @param alignment The alignment of the allocation.
   * @param stream The CUDA stream on which to perform the allocation.
   * @return void* Pointer to the newly allocated memory.
   */
  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    return this->base_type::allocate(stream, bytes);
  }

  /**
   * @brief Deallocates memory pointed to by \p ptr asynchronously.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation.
   * @param alignment The alignment that was passed to the allocate call.
   * @param stream The CUDA stream on which to perform the deallocation.
   */
  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
    this->base_type::deallocate(stream, ptr, bytes);
  }
};

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_async_resource_ref = cuda::mr::async_resource_ref<cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_resource_ref = cuda::mr::resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_async_resource_ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_resource_ref =
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_async_resource_ref =
  cuda::mr::async_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Convert pointer to memory resource into `device_async_resource_ref`, checking for
 * `nullptr`
 *
 * @tparam Resource The type of the memory resource.
 * @param res A pointer to the memory resource.
 * @return A `device_async_resource_ref` to the memory resource.
 * @throws std::logic_error if the memory resource pointer is null.
 */
template <class Resource>
device_async_resource_ref to_device_async_resource_ref_checked(Resource* res)
{
  RMM_EXPECTS(res, "Unexpected null resource pointer.");
  return device_async_resource_ref{*res};
}

/** @} */  // end of group
}  // namespace RMM_NAMESPACE
