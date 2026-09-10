/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <new>
#include <string>
#include <utility>

RMM_NAMESPACE_BEGIN
namespace mr {

pinned_host_memory_resource::pinned_host_memory_resource(host_memory_initializer_t initializer)
  : initializer_{std::move(initializer)}
{
}

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
  if (initializer_) {
    void* ptr{nullptr};
    try {
      ptr = ::operator new(bytes, std::align_val_t{alloc_alignment});
    } catch (std::bad_alloc const& e) {
      auto const msg = std::string{"Failed to allocate "} + rmm::detail::format_bytes(bytes) +
                       std::string{" of host memory: "} + e.what();
      RMM_FAIL(msg.c_str(), rmm::out_of_memory);
    }

    try {
      initializer_(ptr, bytes);
      RMM_CUDA_TRY_ALLOC(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault), bytes);
    } catch (...) {
      ::operator delete(ptr, std::align_val_t{alloc_alignment});
      throw;
    }
    return ptr;
  }

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
  if (initializer_) {
    if (ptr != nullptr) {
      auto const status = cudaHostUnregister(ptr);
      // Do not free memory that CUDA may still consider registered.
      assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
      if (status == cudaSuccess) { ::operator delete(ptr, std::align_val_t{alloc_alignment}); }
    }
    return;
  }

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

bool pinned_host_memory_resource::operator==(
  pinned_host_memory_resource const& other) const noexcept
{
  return static_cast<bool>(initializer_) == static_cast<bool>(other.initializer_);
}

bool pinned_host_memory_resource::operator!=(
  pinned_host_memory_resource const& other) const noexcept
{
  return !(*this == other);
}

}  // namespace mr
RMM_NAMESPACE_END
