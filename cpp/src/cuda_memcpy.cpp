/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/cuda_memcpy.hpp>

RMM_NAMESPACE_BEGIN
namespace detail {

cudaError_t memcpy_async(void* dst, void const* src, std::size_t count, cuda_stream_view stream)
{
  if (count == 0) { return cudaSuccess; }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  if (!stream.is_default()) {
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
    std::size_t attr_idx = 0;
    return cudaMemcpyBatchAsync(&dst, &src, &count, 1, &attrs, &attr_idx, 1, stream.value());
  }
#endif

  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream.value());
}

}  // namespace detail
RMM_NAMESPACE_END
