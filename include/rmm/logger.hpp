/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <rmm/detail/export.hpp>
#include <rmm/detail/format.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <string>

namespace RMM_NAMESPACE {

namespace detail {

/**
 * @brief Returns the default log filename for the RMM global logger.
 *
 * If the environment variable `RMM_DEBUG_LOG_FILE` is defined, its value is used as the path and
 * name of the log file. Otherwise, the file `rmm_log.txt` in the current working directory is used.
 *
 * @return std::string The default log file name.
 */
inline std::string default_log_filename()
{
  auto* filename = std::getenv("RMM_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"rmm_log.txt"} : std::string{filename};
}

/**
 * @brief Simple wrapper around a spdlog::logger that performs RMM-specific initialization
 */
struct logger_wrapper {
  spdlog::logger logger_;  ///< The underlying logger

  logger_wrapper()
    : logger_{"RMM",
              std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                default_log_filename(), true  // truncate file
                )}
  {
    logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    logger_.flush_on(spdlog::level::warn);
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    logger_.info("----- RMM LOG BEGIN [PTDS ENABLED] -----");
#else
    logger_.info("----- RMM LOG BEGIN [PTDS DISABLED] -----");
#endif
    logger_.flush();
#endif
  }
};

inline spdlog::logger& logger()
{
  static detail::logger_wrapper wrapped{};
  return wrapped.logger_;
}
}  // namespace detail

/**
 * @brief Returns the global RMM logger
 *
 * @ingroup logging
 *
 * This is a spdlog logger. The easiest way to log messages is to use the `RMM_LOG_*` macros.
 *
 * @return spdlog::logger& The logger.
 */
[[deprecated(
  "Support for direct access to spdlog loggers in rmm is planned for "
  "removal")]] RMM_EXPORT inline spdlog::logger&
logger()
{
  return detail::logger();
}

//! @cond Doxygen_Suppress
//
// The default is INFO, but it should be used sparingly, so that by default a log file is only
// output if there is important information, warnings, errors, and critical failures
// Log messages that require computation should only be used at level TRACE and DEBUG
#define RMM_LOG_TRACE(...) \
  SPDLOG_LOGGER_TRACE(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))
#define RMM_LOG_DEBUG(...) \
  SPDLOG_LOGGER_DEBUG(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))
#define RMM_LOG_INFO(...) \
  SPDLOG_LOGGER_INFO(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))
#define RMM_LOG_WARN(...) \
  SPDLOG_LOGGER_WARN(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))
#define RMM_LOG_ERROR(...) \
  SPDLOG_LOGGER_ERROR(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))
#define RMM_LOG_CRITICAL(...) \
  SPDLOG_LOGGER_CRITICAL(&rmm::detail::logger(), rmm::detail::formatted_log(__VA_ARGS__))

//! @endcond

}  // namespace RMM_NAMESPACE

//! @endcond
