/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/system_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <new>
#include <string>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

bool is_system_memory_supported(cuda_device_id device_id)
{
  // Check if pageable memory access is supported
  int pageableMemoryAccess;
  RMM_CUDA_TRY(cudaDeviceGetAttribute(
    &pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, device_id.value()));
  return pageableMemoryAccess == 1;
}

}  // namespace detail

system_memory_resource::system_memory_resource()
{
  RMM_EXPECTS(rmm::mr::detail::is_system_memory_supported(rmm::get_current_cuda_device()),
              "System memory allocator is not supported with this hardware/software version.");
}

void* system_memory_resource::allocate([[maybe_unused]] cuda::stream_ref stream,
                                       std::size_t bytes,
                                       std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  RMM_EXPECTS(rmm::is_supported_base_resource_alignment(alignment),
              "Requested alignment is larger than this memory resource supports.",
              rmm::bad_alloc);
  try {
    return rmm::detail::aligned_host_allocate(
      bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, [](std::size_t size) { return ::operator new(size); });
  } catch (std::bad_alloc const& e) {
    auto const msg = std::string("Failed to allocate ") + rmm::detail::format_bytes(bytes) +
                     std::string("of memory: ") + e.what();
    RMM_FAIL(msg.c_str(), rmm::out_of_memory);
  }
}

void system_memory_resource::deallocate(cuda::stream_ref stream,
                                        void* ptr,
                                        std::size_t bytes,
                                        [[maybe_unused]] std::size_t alignment) noexcept
{
  // With `cudaFree`, the CUDA runtime keeps track of dependent operations and does implicit
  // synchronization. However, with SAM, since `free` is immediate, we need to wait for in-flight
  // CUDA operations to finish before freeing the memory, to avoid potential use-after-free errors
  // or race conditions.
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));

  rmm::detail::aligned_host_deallocate(
    ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, [](void* ptr) { ::operator delete(ptr); });
}

void* system_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void system_memory_resource::deallocate_sync(void* ptr,
                                             std::size_t bytes,
                                             std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

bool system_memory_resource::operator==(system_memory_resource const&) const noexcept
{
  return true;
}

bool system_memory_resource::operator!=(system_memory_resource const&) const noexcept
{
  return false;
}

}  // namespace mr
RMM_NAMESPACE_END
