/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>

namespace RMM_EXPORT rmm {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Returns `true` if the process has entered `exit()` / atexit handler execution.
 *
 * Destructors of static objects, as well as atexit handlers registered by other DSOs, can run
 * after `main()` has returned. At that point calling into the CUDA runtime or driver is
 * undefined behavior: the primary context may already be destroyed, and CUDA API calls may
 * dereference released state and crash inside libcuda rather than returning an error.
 *
 * @par Memory resource author contract
 * All RMM memory resources must be safe to destroy at process shutdown. An MR destructor may
 * run during normal program flow (when calling CUDA APIs is safe) or after `exit()` has been
 * called (when it is not). Authors must satisfy this by either:
 *
 * 1. Never calling CUDA APIs from the destructor at all, or
 * 2. Consulting `rmm::process_is_exiting()` in the destructor (and in any helper invoked by
 *    the destructor, such as a `release()` method) and skipping CUDA API calls when it
 *    returns `true`. In that case, resources that would have been explicitly released should
 *    simply be leaked; the OS reclaims them when the process exits.
 *
 * Calling `rmm::process_is_exiting()` from a resource destructor is always safe: it performs a
 * single atomic load (acquire semantics) and never calls into CUDA.
 *
 * @par Example
 * @code
 * class my_resource final : public ... {
 *   ~my_resource() override
 *   {
 *     if (!rmm::process_is_exiting()) {
 *       RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr_));
 *     }
 *   }
 * };
 * @endcode
 *
 * @return `true` if `exit()` has begun; `false` otherwise.
 */
bool process_is_exiting() noexcept;

/** @} */  // end of group

}  // namespace RMM_EXPORT rmm
