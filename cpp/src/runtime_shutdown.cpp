/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/process_is_exiting.hpp>

#include <atomic>
#include <cstdlib>
#include <mutex>

namespace RMM_NAMESPACE {

namespace {

std::atomic<bool> exiting{false};
std::once_flag registered;

}  // namespace

bool process_is_exiting() noexcept { return exiting.load(std::memory_order_acquire); }

namespace detail {

void register_process_exit_hook() noexcept
{
  // The C++ standard guarantees that if the completion of the initialization of a static object A
  // is sequenced-before a call to std::atexit(F), then F runs before A's destructor at
  // termination. RMM exploits this by registering this flag-setter immediately after the internal
  // per-device resource map is constructed, so the flag is observed as true during that map's
  // destructor.
  std::call_once(registered, []() {
    auto const registration_status =
      std::atexit([]() noexcept { exiting.store(true, std::memory_order_release); });
    RMM_EXPECTS(registration_status == 0, "Unable to register process-exit hook");
  });
}

}  // namespace detail
}  // namespace RMM_NAMESPACE
