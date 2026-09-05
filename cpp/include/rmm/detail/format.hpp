/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>

#include <array>
#include <cstdio>
#include <sstream>
#include <string>

RMM_NAMESPACE_BEGIN
namespace detail {

// Stringify a size in bytes to a human-readable value
inline std::string format_bytes(std::size_t value)
{
  static std::array units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

  std::size_t index = 0;
  auto size         = static_cast<double>(value);
  while (size > 1024) {
    size /= 1024;
    index++;
  }

  return std::to_string(size) + ' ' + units.at(index);
}

// Stringify a stream ID
inline std::string format_stream(cuda::stream_ref stream)
{
  std::stringstream sstr{};
  sstr << std::hex << stream.get();
  return sstr.str();
}

}  // namespace detail
RMM_NAMESPACE_END
