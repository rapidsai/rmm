/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
