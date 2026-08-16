/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/export.hpp>

#include <cuda/stream>

RMM_NAMESPACE_BEGIN
namespace detail {

[[nodiscard]] RMM_EXPORT bool is_default_stream(cuda::stream_ref stream) noexcept;

}  // namespace detail
RMM_NAMESPACE_END
