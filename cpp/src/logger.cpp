/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
