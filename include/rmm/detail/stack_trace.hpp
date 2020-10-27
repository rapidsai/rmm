/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// execinfo is a linux-only library, so stack traces will only be available on
// linux systems.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define ENABLE_STACK_TRACES
#endif

#include <memory>
#include <sstream>
#include <vector>

#if defined(ENABLE_STACK_TRACES)
#include <execinfo.h>

namespace rmm {

namespace detail {

/**
 * @brief stack_trace is a class that will capture a stack on instatiation for output later.
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
    // store off a stack for this allocation
    const int MaxStackDepth = 64;
    void* stack[MaxStackDepth];
    auto const depth = backtrace(stack, MaxStackDepth);
    stack_ptrs.insert(stack_ptrs.end(), &stack[0], &stack[depth]);
  }

  friend std::ostream& operator<<(std::ostream& os, const stack_trace& st)
  {
    std::unique_ptr<char*, decltype(&::free)> strings(
      backtrace_symbols(st.stack_ptrs.data(), st.stack_ptrs.size()), &::free);
    if (strings.get() == nullptr) {
      os << "But no stack trace could be found!" << std::endl;
    } else {
      ///@todo: support for demangling of C++ symbol names
      for (int i = 0; i < st.stack_ptrs.size(); ++i) {
        os << "#" << i << " in " << strings.get()[i] << std::endl;
      }
    }
    return os;
  };

 private:
  std::vector<void*> stack_ptrs;
};

}  // namespace detail

}  // namespace rmm
#else
// provide an empty implementation so code doesn't need to know if stack traces
// are enabled
namespace rmm {

namespace detail {

/**
 * @brief stack_trace is a class that will capture a stack on instatiation for output later.
 * It can then be used in an output stream to display stack information.
 *
 * rmm::detail::stack_trace saved_stack;
 *
 * std::cout << "callstack: " << saved_stack;
 *
 */
class stack_trace {
 public:
  stack_trace() {}

  friend std::ostream& operator<<(std::ostream& os, const stack_trace& st)
  {
    os << "stack traces disabled" << std::endl;
    return os;
  };
};

}  // namespace detail

}  // namespace rmm

#endif  // defined(ENABLE_STACK_TRACES)
