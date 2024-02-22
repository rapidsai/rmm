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

#include <cuda/memory_resource>
#include <cuda/std/type_traits>

namespace rmm::mr {

/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief Concept to check whether a resource is a resource adaptor by checking for
 * `get_upstream_resource`.
 */
template <class Resource, class = void>
inline constexpr bool is_resource_adaptor = false;

template <class Resource>
inline constexpr bool is_resource_adaptor<
  Resource,
  cuda::std::void_t<decltype(cuda::std::declval<Resource>().get_upstream_resource())>> =
  cuda::mr::resource<Resource>;

/** @} */  // end of group
}  // namespace rmm::mr
