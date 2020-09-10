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
 *
 * The code in this file has been adapted from the range-v3 library:
 * https://github.com/ericniebler/range-v3/
 * which contains the following copyright notice:
 *
 * Range v3 library
 *
 *  Copyright Eric Niebler 2013-present
 *  Copyright Casey Carter 2016
 *
 *  Use, modification and distribution is subject to the
 *  Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#pragma once

/**
 * @brief This file defines macros for fine-grained compiler diagnostic control.
 *
 * Example usage:
 * ```
 * // A deprecated API:
 * [[deprecated]] void foo() {}
 *
 * foo(); // calling it produces a "deprecated" warning
 *
 * RMM_DIAGNOSTIC_PUSH
 * RMM_DIAGNOSTIC_IGNORE_DEPRECATED_DECLARATIONS
 * foo(); // no deprecation warning!
 * RMM_DIAGNOSTIC_POP
 * ```
 */

#if defined(_MSC_VER) && !defined(__clang__)

#define RMM_DIAGNOSTIC_PUSH __pragma(warning(push))
#define RMM_DIAGNOSTIC_POP __pragma(warning(pop))
#define RMM_DIAGNOSTIC_IGNORE_PRAGMAS __pragma(warning(disable : 4068))
#define RMM_DIAGNOSTIC_IGNORE(X)                               \
  RMM_DIAGNOSTIC_IGNORE_PRAGMAS __pragma(warning(disable : X))
#define RMM_DIAGNOSTIC_IGNORE_DEPRECATED_DECLARATIONS RMM_DIAGNOSTIC_IGNORE(4996)

#elif defined(__GNUC__) || defined(__clang__)

#define RMM_PRAGMA(X) _Pragma(#X)
#define RMM_DIAGNOSTIC_PUSH RMM_PRAGMA(GCC diagnostic push)
#define RMM_DIAGNOSTIC_POP RMM_PRAGMA(GCC diagnostic pop)
#define RMM_DIAGNOSTIC_IGNORE_PRAGMAS RMM_PRAGMA(GCC diagnostic ignored "-Wpragmas")
#define RMM_DIAGNOSTIC_IGNORE(X)                                   \
  RMM_DIAGNOSTIC_IGNORE_PRAGMAS                                    \
  RMM_PRAGMA(GCC diagnostic ignored "-Wunknown-pragmas")           \
    RMM_PRAGMA(GCC diagnostic ignored "-Wunknown-warning-option")  \
    RMM_PRAGMA(GCC diagnostic ignored X)
#define RMM_DIAGNOSTIC_IGNORE_DEPRECATED_DECLARATIONS  \
  RMM_DIAGNOSTIC_IGNORE("-Wdeprecated-declarations")

#else

#define RMM_DIAGNOSTIC_PUSH
#define RMM_DIAGNOSTIC_POP
#define RMM_DIAGNOSTIC_IGNORE_PRAGMAS
#define RMM_DIAGNOSTIC_IGNORE(X)
#define RMM_DIAGNOSTIC_IGNORE_DEPRECATED_DECLARATIONS

#endif
