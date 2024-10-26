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

// TODO: Once we move to proper impl
// #include <rmm/rapids_logger.hpp>
#define SUPPORTS_LOGGING

#ifdef SUPPORTS_LOGGING

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace LOGGER_NAMESPACE {

namespace detail {

/**
 * @brief Returns the default log filename for the RMM global logger.
 *
 * If the environment variable `RAPIDS_DEBUG_LOG_FILE` is defined, its value is used as the path and
 * name of the log file. Otherwise, the file `rmm_log.txt` in the current working directory is used.
 *
 * @return std::string The default log file name.
 */
inline std::string default_log_filename()
{
  auto* filename = std::getenv("RAPIDS_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"rapids_log.txt"} : std::string{filename};
}

/**
 * @struct impl
 * @brief The real implementation of the logger using spdlog with a basic file sink.
 */
struct impl {
  spdlog::logger underlying;  ///< spdlog logger instance

  /**
   * @brief Constructor for the real implementation of the logger.
   * Initializes the logger with a basic file sink.
   */
  impl(std::string name, std::string filename)
    : underlying{
        name,
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true  // truncate file
                                                            )}
  {
    underlying.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    underlying.flush_on(spdlog::level::warn);
  }

  void log(level_enum lvl, const std::string& message)
  {
    underlying.log(static_cast<spdlog::level::level_enum>(static_cast<int32_t>(lvl)), message);
  }
};

/**
 * @brief Represent a size in number of bytes.
 */
struct bytes {
  std::size_t value;  ///< The size in bytes

  /**
   * @brief Construct a new bytes object
   *
   * @param os The output stream
   * @param value The size in bytes
   */
  friend std::ostream& operator<<(std::ostream& os, bytes const& value)
  {
    static std::array units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

    int index = 0;
    auto size = static_cast<double>(value.value);
    while (size > 1024) {
      size /= 1024;
      index++;
    }
    return os << size << ' ' << units.at(index);
  }
};

}  // namespace detail

// TODO: Probably don't want to default construct here for the underlying spdlog logger.
inline logger::logger(std::string name, std::string filename)
  : pImpl{std::make_unique<detail::impl>(name, filename)}
{
}

inline logger::~logger() = default;

inline void logger::log(level_enum lvl, std::string const& message) { pImpl->log(lvl, message); }

// TODO: The detail implementations are just for backwards compat of the spdlog
// version and should be removed.
namespace detail {

inline class logger& default_logger()
{
  static class logger logger_ = [] {
    class logger logger_ {
      "RAPIDS", detail::default_log_filename()
    };
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    logger_.info("----- RAPIDS LOG BEGIN [PTDS ENABLED] -----");
#else
    logger_.info("----- RAPIDS LOG BEGIN [PTDS DISABLED] -----");
#endif
#endif
    return logger_;
  }();
  return logger_;
}

inline class spdlog::logger& logger() { return default_logger().pImpl->underlying; }

}  // namespace detail

// TODO: We can't macro this comment, maybe another reason to use configure_file.
}  // namespace LOGGER_NAMESPACE

// Doxygen doesn't like this because we're overloading something from fmt
//! @cond Doxygen_Suppress
template <>
struct fmt::formatter<rmm::detail::bytes> : fmt::ostream_formatter {};

#endif
