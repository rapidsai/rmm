/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {

void* managed_memory_resource::allocate([[maybe_unused]] cuda::stream_ref stream,
                                        std::size_t bytes,
                                        std::size_t alignment)
{
  // FIXME: Unlike cudaMalloc, cudaMallocManaged will throw an error for 0
  // size allocations.
  if (bytes == 0) { return nullptr; }
  RMM_EXPECTS(rmm::is_supported_base_resource_alignment(alignment),
              "Requested alignment is larger than this memory resource supports.",
              rmm::bad_alloc);

  void* ptr{nullptr};
  RMM_CUDA_TRY_ALLOC(cudaMallocManaged(&ptr, bytes), bytes);
  return ptr;
}

void managed_memory_resource::deallocate([[maybe_unused]] cuda::stream_ref stream,
                                         void* ptr,
                                         [[maybe_unused]] std::size_t bytes,
                                         [[maybe_unused]] std::size_t alignment) noexcept
{
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr));
}

void* managed_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void managed_memory_resource::deallocate_sync(void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

bool managed_memory_resource::operator==(managed_memory_resource const&) const noexcept
{
  return true;
}

bool managed_memory_resource::operator!=(managed_memory_resource const&) const noexcept
{
  return false;
}

}  // namespace mr
RMM_NAMESPACE_END
