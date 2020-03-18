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

#include "nvtx3.hpp"

namespace rmm {

/**
 * @brief Tag type for libcudf's NVTX domain.
 *
 */
struct librmm_domain {
  static constexpr char const* name{"librmm"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 *
 */
using thread_range = ::nvtx3::domain_thread_range<librmm_domain>;

/**
 * @brief Color for the RMM NVTX domain
 * 
 */
static constexpr nvtx3::rgb librmm_color{205, 22, 75};

}  // namespace cudf

#define RMM_FUNC_RANGE_IN(D)                                                             \
  static ::nvtx3::registered_message<D> const nvtx3_func_name__{__func__};               \
  static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__, rmm::librmm_color}; \
  ::nvtx3::domain_thread_range<D> const nvtx3_range__{nvtx3_func_attr__};

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
 *
 */
#define RMM_FUNC_RANGE() RMM_FUNC_RANGE_IN(rmm::librmm_domain)
