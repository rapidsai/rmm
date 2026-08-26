/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/cuda_stream.hpp>
#include <rmm/detail/export.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace detail {

[[nodiscard]] RMM_EXPORT cudaError_t memcpy_async(void* dst,
                                                  void const* src,
                                                  std::size_t count,
                                                  cuda::stream_ref stream);

}  // namespace detail
RMM_NAMESPACE_END
