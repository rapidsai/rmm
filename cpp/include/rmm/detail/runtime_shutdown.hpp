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
 * This registers the single process-exit hook used to make resources held in RMM's internal
 * per-device resource map safe to destruct during process termination. It is not a general
 * per-static-object registration facility.
 */
RMM_EXPORT void register_process_exit_hook() noexcept;

}  // namespace detail
}  // namespace RMM_NAMESPACE
