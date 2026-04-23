/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cstdlib>

/*
 * Standalone test that verifies rmm::process_is_exiting() reports true while the C runtime is
 * executing atexit handlers. The fix in rapidsai/rmm#2367 depends on this behavior: resource
 * destructors running during exit() must observe the flag as set so that they can skip CUDA
 * calls.
 *
 * The C++ standard guarantees that if the completion of the initialization of a static object A
 * is sequenced-before a call to std::atexit(F), then F runs before A's destructor at
 * termination. RMM exploits this by having register_process_exit_hook() register a flag-setter
 * via std::atexit on first use; it is called from get_ref_map() immediately after the static
 * map is constructed, so the flag is observed as true during that map's destructor.
 *
 * This binary exercises the same mechanism without any per-device map or CUDA involvement:
 *
 *   1. main() registers `checker` via std::atexit FIRST.
 *   2. main() then calls register_process_exit_hook(), which registers its internal
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
  if (!rmm::process_is_exiting()) { std::_Exit(1); }
}

}  // namespace

int main()
{
  // Register the checker before rmm's flag-setter so that the checker runs AFTER the
  // flag-setter at termination.
  if (std::atexit(checker) != 0) { return 2; }
  rmm::detail::register_process_exit_hook();
  return 0;
}
