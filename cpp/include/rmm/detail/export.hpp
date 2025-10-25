/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Macros used for defining symbol visibility, only GLIBC is supported
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_EXPORT    __attribute__((visibility("default")))
#define RMM_HIDDEN    __attribute__((visibility("hidden")))
#define RMM_NAMESPACE RMM_EXPORT rmm
#else
#define RMM_EXPORT
#define RMM_HIDDEN
#define RMM_NAMESPACE rmm
#endif
