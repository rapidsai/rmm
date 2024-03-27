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

// Only include <rmm/logger.hpp> if needed in RMM_LOGGING_ASSERT below. The
// logger can be extremely expensive to compile, so we want to avoid including
// it.
#if !defined(NDEBUG)
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>

#include <cassert>
#endif

/**
 * @brief Assertion that logs a CRITICAL log message on failure.
 */
#ifdef NDEBUG
#define RMM_LOGGING_ASSERT(_expr) (void)0
#elif SPDLOG_ACTIVE_LEVEL < SPDLOG_LEVEL_OFF
#define RMM_LOGGING_ASSERT(_expr)                                                                 \
  do {                                                                                            \
    bool const success = (_expr);                                                                 \
    if (!success) {                                                                               \
      RMM_LOG_CRITICAL(                                                                           \
        "[" __FILE__ ":" RMM_STRINGIFY(__LINE__) "] Assertion " RMM_STRINGIFY(_expr) " failed."); \
      rmm::logger().flush();                                                                      \
      /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */                   \
      assert(success);                                                                            \
    }                                                                                             \
  } while (0)
#else
#define RMM_LOGGING_ASSERT(_expr) assert((_expr));
#endif
