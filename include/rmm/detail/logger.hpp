/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// If using GCC, temporary workaround for older libcudacxx defining _LIBCPP_VERSION
// undefine it before including spdlog, due to fmtlib checking if it is defined
// TODO: remove once libcudacxx is on Github and RAPIDS depends on it
#ifdef __GNUG__
#undef _LIBCPP_VERSION
#endif
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

namespace rmm {

namespace detail {

inline std::string default_log_filename()
{
  auto filename = std::getenv("RMM_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"rmm_log.txt"} : std::string{filename};
}

// Simple wrapper around a spdlog::logger that performs RMM-specific initialization
struct logger_wrapper {
  spdlog::logger logger_;

  logger_wrapper()
    : logger_{"RMM",
              std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                default_log_filename(), true  // truncate file
                )}
  {
    logger_.set_pattern("[%l][%t][%H:%M:%S:%f] %v");
    logger_.flush_on(spdlog::level::warn);

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    logger_.info("----- RMM LOG BEGIN [PTDS ENABLED] -----");
#else
    logger_.info("----- RMM LOG BEGIN [PTDS DISABLED] -----");
#endif
    logger_.flush();
  }
};

}  // namespace detail

inline spdlog::logger& logger()
{
  static detail::logger_wrapper w{};
  return w.logger_;
}

#define RMM_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(&rmm::logger(), __VA_ARGS__)
#define RMM_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(&rmm::logger(), __VA_ARGS__)
#define RMM_LOG_INFO(...) SPDLOG_LOGGER_INFO(&rmm::logger(), __VA_ARGS__)
#define RMM_LOG_WARN(...) SPDLOG_LOGGER_WARN(&rmm::logger(), __VA_ARGS__)
#define RMM_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(&rmm::logger(), __VA_ARGS__)
#define RMM_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(&rmm::logger(), __VA_ARGS__)

}  // namespace rmm
