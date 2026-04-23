/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>

namespace RMM_NAMESPACE {
namespace detail {

/**
 * @brief Register the atexit callback that flips the flag observed by `rmm::process_is_exiting()`.
 *
 * Idempotent: safe to call from multiple places; only the first call registers the callback.
 * Callers must invoke this after constructing any static object whose destructor may run during
 * process exit and needs to consult `rmm::process_is_exiting()`. The C++ standard guarantees
 * that if a static object's construction is sequenced-before a `std::atexit` call, the atexit
 * callback runs before that object's destructor at termination.
 */
RMM_EXPORT void register_process_exit_hook() noexcept;

}  // namespace detail
}  // namespace RMM_NAMESPACE
