/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace detail {

[[nodiscard]] RMM_EXPORT cudaError_t memcpy_async(void* dst,
                                                  void const* src,
                                                  std::size_t count,
                                                  cuda_stream_view stream);

}  // namespace detail
RMM_NAMESPACE_END
