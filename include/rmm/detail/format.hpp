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
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace RMM_NAMESPACE {
namespace detail {

/**
 * @brief Format a message string with printf-style formatting
 *
 * This function performs printf-style formatting to avoid the need for fmt
 * or spdlog's own templated APIs (which would require exposing spdlog
 * symbols publicly) and returns the formatted message as a `std::string`.
 *
 * @param format The format string
 * @param args The format arguments
 * @return The formatted message
 * @throw rmm::logic_error if an error occurs during formatting
 */
template <typename... Args>
std::string formatted_log(std::string const& format, Args&&... args)
{
  auto convert_to_c_string = [](auto&& arg) -> decltype(auto) {
    using ArgType = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<ArgType, std::string>) {
      return arg.c_str();
    } else {
      return std::forward<decltype(arg)>(arg);
    }
  };

  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
  auto retsize =
    std::snprintf(nullptr, 0, format.c_str(), convert_to_c_string(std::forward<Args>(args))...);
  RMM_EXPECTS(retsize >= 0, "Error during formatting.");
  if (retsize == 0) { return {}; }
  auto size = static_cast<std::size_t>(retsize) + 1;  // for null terminator
  // NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), convert_to_c_string(std::forward<Args>(args))...);
  // NOLINTEND(cppcoreguidelines-pro-type-vararg)
  return {buf.get(), buf.get() + size - 1};  // drop '\0'
}

// specialization for no arguments
template <>
inline std::string formatted_log(std::string const& format)
{
  return format;
}

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
