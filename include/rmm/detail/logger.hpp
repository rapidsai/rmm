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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <string>

namespace rmm {

namespace detail {

inline std::string get_default_log_filename()
{
  auto filename = std::getenv("RMM_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"rmm_debug_log.txt"} : std::string{filename};
}

inline spdlog::logger& logger()
{
  static spdlog::logger logger_("RMM",
                                std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                                  get_default_log_filename(), true /*truncate file*/));
  return logger_;
}

}  // namespace detail

}  // namespace rmm
