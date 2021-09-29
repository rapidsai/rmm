/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace rmm::detail {

/// Gets the available and total device memory in bytes for the current device
inline std::pair<std::size_t, std::size_t> available_device_memory()
{
  std::size_t free{};
  std::size_t total{};
  RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
  return {free, total};
}

}  // namespace rmm::detail
