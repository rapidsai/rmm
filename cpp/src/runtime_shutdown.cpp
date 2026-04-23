/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/process_is_exiting.hpp>

#include <atomic>
#include <cstdlib>

namespace RMM_NAMESPACE {

namespace {

std::atomic<bool>& exiting_flag() noexcept
{
  static std::atomic<bool> flag{false};
  return flag;
}

}  // namespace

bool process_is_exiting() noexcept { return exiting_flag().load(std::memory_order_acquire); }

namespace detail {

void register_process_exit_hook() noexcept
{
  // The C++ standard guarantees that if a static object's construction is sequenced-before a
  // call to std::atexit, the atexit callback runs before that object's destructor at
  // termination (see https://en.cppreference.com/cpp/utility/program/exit). Callers must
  // therefore invoke this function after constructing the static object whose destructor needs
  // to observe the exit flag.
  static std::atomic<bool> registered{false};
  bool expected = false;
  if (registered.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    std::atexit([]() noexcept { exiting_flag().store(true, std::memory_order_release); });
  }
}

}  // namespace detail
}  // namespace RMM_NAMESPACE
