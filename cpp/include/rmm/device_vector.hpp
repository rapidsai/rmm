/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/thrust_allocator_adaptor.hpp>

#include <thrust/device_vector.h>

namespace RMM_NAMESPACE {
/**
 * @addtogroup thrust_integrations
 * @{
 * @file
 */
/**
 * @brief Alias for a thrust::device_vector that uses RMM for memory allocation.
 *
 */
template <typename T>
using device_vector = thrust::device_vector<T, rmm::mr::thrust_allocator<T>>;

/** @} */  // end of group
}  // namespace RMM_NAMESPACE
