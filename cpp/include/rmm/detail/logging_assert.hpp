/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
#elif RMM_LOG_ACTIVE_LEVEL < RMM_LOG_LEVEL_OFF
#define RMM_LOGGING_ASSERT(_expr)                                                                 \
  do {                                                                                            \
    bool const success = (_expr);                                                                 \
    if (!success) {                                                                               \
      RMM_LOG_CRITICAL(                                                                           \
        "[" __FILE__ ":" RMM_STRINGIFY(__LINE__) "] Assertion " RMM_STRINGIFY(_expr) " failed."); \
      rmm::default_logger().flush();                                                              \
      /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */                   \
      assert(success);                                                                            \
    }                                                                                             \
  } while (0)
#else
#define RMM_LOGGING_ASSERT(_expr) assert((_expr));
#endif
