/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <array>
#include <cstdio>
#include <sstream>
#include <string>

namespace RMM_NAMESPACE {
namespace detail {

// Stringify a size in bytes to a human-readable value
inline std::string format_bytes(std::size_t value)
{
  static std::array units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

  int index = 0;
  auto size = static_cast<double>(value);
  while (size > 1024) {
    size /= 1024;
    index++;
  }

  return std::to_string(size) + ' ' + units.at(index);
}

// Stringify a stream ID
inline std::string format_stream(rmm::cuda_stream_view stream)
{
  std::stringstream sstr{};
  sstr << std::hex << stream.value();
  return sstr.str();
}

}  // namespace detail
}  // namespace RMM_NAMESPACE
