/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

namespace rmm::test {

/**
 * @brief Should tests be skipped if cudaMallocAsync is not supported with the CUDA driver/runtime
 * version?
 */
inline bool should_skip_async()
{
  static auto runtime_version{[] {
    int runtime_version{};
    RMM_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
    return runtime_version;
  }()};
  static auto driver_version{[] {
    int driver_version{};
    RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
    return driver_version;
  }()};
  constexpr auto min_async_version{11020};
  return runtime_version < min_async_version || driver_version < min_async_version;
}

}  // namespace rmm::test
