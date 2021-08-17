/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda/memory_resource>

#include <cstddef>
#include <utility>

namespace rmm {
namespace mr {

namespace experimental {  // to avoid conflicts with existing memory resources

/**
 * @brief Base class for all libcudf device memory allocation.
 *
 * This class serves as the interface that all custom device memory
 * implementations must satisfy.
 */
// using device_memory_resource = cuda::stream_ordered_memory_resource<cuda::memory_kind::device>;

// using device_resource_view = cuda::stream_ordered_resource_view<cuda::memory_access::device>;

}  // namespace experimental
}  // namespace mr
}  // namespace rmm
