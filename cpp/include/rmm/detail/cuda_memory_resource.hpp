/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#ifndef LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#error \
  "RMM requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined. Please add -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to the compiler flags (this is done automatically when using RMM via CMake)."
#endif

#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>

namespace RMM_NAMESPACE {
namespace detail {
namespace polyfill {

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
template <class Resource>
inline constexpr bool resource = cuda::mr::synchronous_resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool resource_with = cuda::mr::synchronous_resource_with<Resource, Properties...>;
template <class Resource>
inline constexpr bool async_resource = cuda::mr::resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool async_resource_with = cuda::mr::resource_with<Resource, Properties...>;
#else   // ^^^ CCCL >= 3.1 ^^^ / vvv CCCL < 3.1 vvv
template <class Resource>
inline constexpr bool resource = cuda::mr::resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool resource_with = cuda::mr::resource_with<Resource, Properties...>;
template <class Resource>
inline constexpr bool async_resource = cuda::mr::async_resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool async_resource_with = cuda::mr::async_resource_with<Resource, Properties...>;
#endif  // CCCL < 3.1

}  // namespace polyfill
}  // namespace detail
}  // namespace RMM_NAMESPACE
