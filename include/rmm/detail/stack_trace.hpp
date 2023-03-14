/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>

// execinfo is a linux-only library, so stack traces will only be available on
// linux systems.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_ENABLE_STACK_TRACES
#endif

#include <sstream>

#if defined(RMM_ENABLE_STACK_TRACES)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <cstddef>
#include <memory>
#include <vector>
#endif

namespace rmm::detail {

/**
 * @brief stack_trace is a class that will capture a stack on instantiation for output later.
 * It can then be used in an output stream to display stack information.
 *
 * rmm::detail::stack_trace saved_stack;
 *
 * std::cout << "callstack: " << saved_stack;
 *
 */
class stack_trace {
 public:
  stack_trace()
  {
#if defined(RMM_ENABLE_STACK_TRACES)
    const int MaxStackDepth = 64;
    std::array<void*, MaxStackDepth> stack{};
    auto const depth = backtrace(stack.begin(), MaxStackDepth);
    stack_ptrs.insert(stack_ptrs.end(), stack.begin(), stack.begin() + depth);
#endif  // RMM_ENABLE_STACK_TRACES
  }

  friend std::ostream& operator<<(std::ostream& os, const stack_trace& trace)
  {
#if defined(RMM_ENABLE_STACK_TRACES)
    std::unique_ptr<char*, decltype(&::free)> strings(
      backtrace_symbols(trace.stack_ptrs.data(), static_cast<int>(trace.stack_ptrs.size())),
      &::free);

    RMM_EXPECTS(strings != nullptr, "Unexpected null stack trace symbols");
    // Iterate over the stack pointers converting to a string
    for (std::size_t i = 0; i < trace.stack_ptrs.size(); ++i) {
      // Leading index
      os << "#" << i << " in ";

      auto const str = [&] {
        Dl_info info;
        if (dladdr(trace.stack_ptrs[i], &info) != 0) {
          int status = -1;  // Demangle the name. This can occasionally fail

          std::unique_ptr<char, decltype(&::free)> demangled(
            abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status), &::free);
          // If it fails, fallback to the dli_name.
          if (status == 0 or (info.dli_sname != nullptr)) {
            auto const* name = status == 0 ? demangled.get() : info.dli_sname;
            return name + std::string(" from ") + info.dli_fname;
          }
        }
        return std::string(strings.get()[i]);
      }();

      os << str << std::endl;
    }
#else
    os << "stack traces disabled" << std::endl;
#endif  // RMM_ENABLE_STACK_TRACES
    return os;
  };

#if defined(RMM_ENABLE_STACK_TRACES)
 private:
  std::vector<void*> stack_ptrs;
#endif  // RMM_ENABLE_STACK_TRACES
};

}  // namespace rmm::detail
