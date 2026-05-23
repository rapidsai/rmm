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
 * Destructors of static objects, as well as atexit handlers registered by other DSOs, run
 * during process termination after `main()` has returned. At that point calling into the CUDA
 * runtime or driver is undefined behavior: the primary context may already be destroyed, and
 * CUDA API calls may dereference released state and crash inside libcuda rather than returning
 * an error.
 *
 * Use this function from a memory resource destructor (or a helper invoked by a destructor, such
 * as a `release()` method) when the resource may be held in RMM's internal per-device resource
 * map and destroyed during process termination. In that case the destructor may run after the
 * CUDA primary context has been destroyed, and calling into the CUDA runtime is undefined
 * behavior. Destructors can avoid that by:
 *
 * 1. Never calling CUDA APIs from the destructor at all, or
 * 2. Consulting `rmm::process_is_exiting()` in the destructor (and in any helper invoked by
 *    the destructor, such as a `release()` method) and skipping CUDA API calls when it
 *    returns `true`. In that case, resources that would have been explicitly released should be
 *    leaked; the OS reclaims them when the process exits.
 *
 * Storing RMM objects with static or thread-local scope is unsupported. Users should not create
 * their own static containers of RMM objects and rely on `rmm::process_is_exiting()` to make
 * those destructors safe.
 *
 * Calling `rmm::process_is_exiting()` from a resource destructor is always safe: it performs a
 * single atomic load (acquire semantics) and never calls into CUDA.
 *
 * Example:
 * @code{.cpp}
 * class my_resource final : public ... {
 *   ~my_resource() override
 *   {
 *     if (rmm::process_is_exiting()) {
 *       return;
 *     }
 *     RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr_));
 *   }
 * };
 * @endcode
 *
 * @return `true` if `exit()` has begun; `false` otherwise.
 */
bool process_is_exiting() noexcept;

/** @} */  // end of group

}  // namespace RMM_EXPORT rmm
