/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cstdlib>

/*
 * Standalone test that verifies rmm::process_is_exiting() reports true while the C runtime is
 * executing atexit handlers. Resource destructors running during exit() must observe the flag
 * as set so that they can skip CUDA calls.
 *
 * This binary exercises the real integration point without any CUDA involvement:
 *
 *   1. main() registers `checker` via std::atexit FIRST.
 *   2. main() then calls detail::get_ref_map(). This constructs the static per-device map and,
 *      immediately afterward, calls register_process_exit_hook(), which registers the internal
 *      flag-setter via std::atexit SECOND.
 *   3. main() returns 0.
 *   4. At termination, atexit handlers run in LIFO order: first the flag-setter (sets the
 *      flag to true), then `checker` (reads the flag).
 *   5. If `checker` observes the flag as true, the process exits with status 0; otherwise it
 *      calls _Exit(1) to signal failure.
 *
 * CTest checks the exit status.
 */

namespace {

void checker() noexcept
{
  if (!rmm::process_is_exiting()) {
    // The process is already running atexit handlers. Use std::_Exit rather than std::exit to
    // report failure without re-entering normal termination or running more cleanup.
    std::_Exit(1);
  }
}

}  // namespace

int main()
{
  // Register the checker before get_ref_map() so that the checker runs AFTER the flag-setter at
  // termination.
  if (std::atexit(checker) != 0) { return 2; }
  [[maybe_unused]] auto& map = rmm::mr::detail::get_ref_map();
  return 0;
}
