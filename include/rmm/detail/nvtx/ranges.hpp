/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#if defined(RMM_NVTX)
#define RMM_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(rmm::librmm_domain)
#else
#define RMM_FUNC_RANGE()
#endif
