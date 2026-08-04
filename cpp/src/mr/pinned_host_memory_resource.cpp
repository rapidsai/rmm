/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {

void* pinned_host_memory_resource::allocate([[maybe_unused]] cuda::stream_ref stream,
                                            std::size_t bytes,
                                            std::size_t alignment)
{
  // don't allocate anything if the user requested zero bytes
  if (0 == bytes) { return nullptr; }
  RMM_EXPECTS(rmm::is_supported_base_resource_alignment(alignment),
              "Requested alignment is larger than this memory resource supports.",
              rmm::bad_alloc);

  std::size_t constexpr alloc_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  return rmm::detail::aligned_host_allocate(bytes, alloc_alignment, [](std::size_t size) {
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, size, cudaHostAllocDefault), size);
    return ptr;
  });
}

void pinned_host_memory_resource::deallocate([[maybe_unused]] cuda::stream_ref stream,
                                             void* ptr,
                                             std::size_t bytes,
                                             [[maybe_unused]] std::size_t alignment) noexcept
{
  std::size_t constexpr alloc_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  rmm::detail::aligned_host_deallocate(ptr, bytes, alloc_alignment, [](void* ptr) {
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFreeHost(ptr));
  });
}

void* pinned_host_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void pinned_host_memory_resource::deallocate_sync(void* ptr,
                                                  std::size_t bytes,
                                                  std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

bool pinned_host_memory_resource::operator==(pinned_host_memory_resource const&) const noexcept
{
  return true;
}

bool pinned_host_memory_resource::operator!=(pinned_host_memory_resource const&) const noexcept
{
  return false;
}

}  // namespace mr
RMM_NAMESPACE_END
