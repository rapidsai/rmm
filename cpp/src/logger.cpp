/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <rmm/logger.hpp>

#include <cstdlib>
#include <memory>
#include <string>

namespace rmm {

rapids_logger::sink_ptr default_sink()
{
  auto* filename = std::getenv("RMM_DEBUG_LOG_FILE");
  if (filename != nullptr) {
    return std::make_shared<rapids_logger::basic_file_sink_mt>(filename, true);
  }
  return std::make_shared<rapids_logger::stderr_sink_mt>();
}

std::string default_pattern() { return "[%6t][%H:%M:%S:%f][%-6l] %v"; }

rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"RMM", {default_sink()}};
    logger_.set_pattern(default_pattern());
    return logger_;
  }();
  return logger_;
}

}  // namespace rmm
