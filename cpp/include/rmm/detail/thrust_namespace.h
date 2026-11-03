/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/export.hpp>

#include <thrust/detail/config.h>  // namespace macros

#ifdef THRUST_WRAPPED_NAMESPACE

// Ensure the namespace exist before we import it
// so that this include can occur before thrust includes
namespace THRUST_WRAPPED_NAMESPACE {
namespace thrust {
}
}  // namespace THRUST_WRAPPED_NAMESPACE

RMM_NAMESPACE_BEGIN
using namespace THRUST_WRAPPED_NAMESPACE;
RMM_NAMESPACE_END

#endif
