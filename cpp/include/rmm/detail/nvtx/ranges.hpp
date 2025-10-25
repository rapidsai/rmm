/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(RMM_NVTX)
#include <nvtx3/nvtx3.hpp>

namespace rmm {
/**
 * @brief Tag type for librmm's NVTX domain.
 */
struct librmm_domain {
  static constexpr char const* name{"librmm"};  ///< Name of the librmm domain
};

/**
 * @brief Alias for an NVTX range in the librmm domain.
 *
 * Customizes an NVTX range with the given input.
 *
 * Example:
 * ```
 * void some_function(){
 *    rmm::scoped_range rng{"custom_name"}; // Customizes range name
 *    ...
 * }
 * ```
 */
using scoped_range = ::nvtx3::scoped_range_in<librmm_domain>;

}  // namespace rmm

/**
 * @brief Convenience macro for generating an NVTX range in the `librmm` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    RMM_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define RMM_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(rmm::librmm_domain)
#else
#define RMM_FUNC_RANGE()
#endif
