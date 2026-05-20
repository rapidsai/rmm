/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/prefetch.hpp>

#include <cuda_runtime_api.h>

namespace rmm {

namespace {
void prefetch_impl(void const* ptr,
                   std::size_t size,
                   rmm::cuda_device_id device,
                   cuda::stream_ref stream)
{
  if (!rmm::detail::concurrent_managed_access::is_supported()) { return; }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  cudaMemLocation location{
    (device.value() == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice,
    device.value()};
  constexpr int flags = 0;
  cudaError_t result  = cudaMemPrefetchAsync(ptr, size, location, flags, stream.get());
#else
  cudaError_t result = cudaMemPrefetchAsync(ptr, size, device.value(), stream.get());
#endif
  // cudaErrorInvalidValue is returned when non-managed memory is passed to
  // cudaMemPrefetchAsync. We treat this as a no-op.
  if (result != cudaErrorInvalidValue && result != cudaSuccess) { RMM_CUDA_TRY(result); }
}
}  // namespace

void prefetch(void const* ptr,
              std::size_t size,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  prefetch_impl(ptr, size, device, cuda::stream_ref{stream});
}

}  // namespace rmm
