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

#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/prefetch.hpp>

#include <cuda_runtime_api.h>

namespace rmm {

void prefetch(void const* ptr,
              std::size_t size,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  if (!rmm::detail::concurrent_managed_access::is_supported()) { return; }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  cudaMemLocation location{
    (device.value() == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice,
    device.value()};
  constexpr int flags = 0;
  cudaError_t result  = cudaMemPrefetchAsync(ptr, size, location, flags, stream.value());
#else
  cudaError_t result = cudaMemPrefetchAsync(ptr, size, device.value(), stream.value());
#endif
  // cudaErrorInvalidValue is returned when non-managed memory is passed to
  // cudaMemPrefetchAsync. We treat this as a no-op.
  if (result != cudaErrorInvalidValue && result != cudaSuccess) { RMM_CUDA_TRY(result); }
}

}  // namespace rmm
