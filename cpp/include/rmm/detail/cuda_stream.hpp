/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/export.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

RMM_NAMESPACE_BEGIN
namespace detail {

/**
 * @brief Indicates whether a stream is a default stream in the current compilation mode.
 *
 * In per-thread default stream mode, the null stream and `cudaStreamPerThread` are default
 * streams. Otherwise, the null stream and `cudaStreamLegacy` are default streams.
 *
 * @param stream The stream to check.
 * @return true if `stream` is a default stream in the current compilation mode.
 */
[[nodiscard, maybe_unused]] static bool is_default_stream(cuda::stream_ref stream) noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return stream.get() == cudaStream_t{} || stream.get() == cudaStream_t{cudaStreamPerThread};
#else
  return stream.get() == cudaStream_t{} || stream.get() == cudaStream_t{cudaStreamLegacy};
#endif
}

}  // namespace detail
RMM_NAMESPACE_END
