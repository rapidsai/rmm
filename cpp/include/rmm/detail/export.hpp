/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Macros used for defining symbol visibility, only GLIBC is supported.
//
// RMM_NAMESPACE vs plain `namespace rmm`:
//
// Use `namespace RMM_NAMESPACE` for symbols that meet ANY of these criteria:
//   1. Compiled into librmm.so (i.e., have a corresponding .cpp file)
//   2. Exception types that may be thrown/caught across DSO boundaries
//   3. Base classes with virtual functions used polymorphically across DSOs
//
// Use plain `namespace rmm` for:
//   1. All template classes (instantiated in downstream libraries)
//   2. Header-only derived classes (compiled into downstream libraries)
//   3. All detail/internal headers
//
// Rationale: RMM_NAMESPACE forces symbols to be publicly exported via
// __attribute__((visibility("default"))), even when downstream libraries
// compile with -fvisibility=hidden. For header-only and template code that
// gets compiled into downstream libraries, this can cause ODR violations
// when multiple libraries (compiled against different RMM versions) are
// loaded together. Using plain `namespace rmm` allows downstream libraries
// to control symbol visibility via their own compile flags.
//
// However, exception types and polymorphic base classes MUST have default
// visibility for cross-DSO exception catching and dynamic_cast to work.
// See GitHub issue #1652 and GCC wiki on visibility for details.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_EXPORT    __attribute__((visibility("default")))
#define RMM_HIDDEN    __attribute__((visibility("hidden")))
#define RMM_NAMESPACE RMM_EXPORT rmm
#else
#define RMM_EXPORT
#define RMM_HIDDEN
#define RMM_NAMESPACE rmm
#endif
