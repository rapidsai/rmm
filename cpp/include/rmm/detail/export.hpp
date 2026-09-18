/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/version_config.hpp>

#include <cuda/version>

#if CCCL_MAJOR_VERSION < 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION < 3)
#error "RMM requires CCCL version 3.3 or newer."
#endif

// Macros used for defining symbol visibility, only GLIBC is supported
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_EXPORT __attribute__((visibility("default")))
#define RMM_HIDDEN __attribute__((visibility("hidden")))
#else
#define RMM_EXPORT
#define RMM_HIDDEN
#endif

// Place RMM symbols in an inline namespace for the RMM major/minor ABI version. This preserves
// the rmm:: source-level API while allowing different RMM ABI versions to coexist in one process.
#define RMM_DETAIL_ABI_NAMESPACE_IMPL(major, minor) _RMM_##major##_##minor
#define RMM_DETAIL_ABI_NAMESPACE(major, minor)      RMM_DETAIL_ABI_NAMESPACE_IMPL(major, minor)
#define RMM_ABI_NAMESPACE                           RMM_DETAIL_ABI_NAMESPACE(RMM_VERSION_MAJOR, RMM_VERSION_MINOR)

#define RMM_NAMESPACE rmm::RMM_ABI_NAMESPACE
#define RMM_NAMESPACE_BEGIN  \
  namespace RMM_EXPORT rmm { \
  inline namespace RMM_ABI_NAMESPACE {
#define RMM_NAMESPACE_END \
  }                       \
  }

// Work around breathe "friend constexpr friend" bug (breathe-doc/breathe#916).
// Doxygen expands this to plain `friend`; normal builds get `constexpr friend`.
#if defined(DOXYGEN)
#define RMM_CONSTEXPR_FRIEND friend
#else
#define RMM_CONSTEXPR_FRIEND constexpr friend
#endif
