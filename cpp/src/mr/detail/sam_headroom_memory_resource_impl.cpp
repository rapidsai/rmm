/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/sam_headroom_memory_resource_impl.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

sam_headroom_memory_resource_impl::sam_headroom_memory_resource_impl(std::size_t headroom)
  : system_mr_{}, headroom_{headroom}
{
}

void* sam_headroom_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t /*alignment*/)
{
  void* pointer = system_mr_.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto const free        = rmm::available_device_memory().first;
  auto const allocatable = free > headroom_ ? free - headroom_ : 0UL;
  auto const gpu_portion =
    rmm::align_down(std::min(allocatable, bytes), rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const cpu_portion = bytes - gpu_portion;

  if (gpu_portion != 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{cudaMemLocationTypeDevice, rmm::get_current_cuda_device().value()};
    RMM_CUDA_TRY(cudaMemAdvise(pointer, gpu_portion, cudaMemAdviseSetPreferredLocation, location));
#else
    RMM_CUDA_TRY(cudaMemAdvise(pointer,
                               gpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               rmm::get_current_cuda_device().value()));
#endif
  }
  if (cpu_portion != 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{cudaMemLocationTypeHost, 0};
    RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(pointer) + gpu_portion,
                               cpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               location));
#else
    RMM_CUDA_TRY(cudaMemAdvise(static_cast<char*>(pointer) + gpu_portion,
                               cpu_portion,
                               cudaMemAdviseSetPreferredLocation,
                               cudaCpuDeviceId));
#endif
  }

  return pointer;
}

void sam_headroom_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t /*alignment*/) noexcept
{
  system_mr_.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void* sam_headroom_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void sam_headroom_memory_resource_impl::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
