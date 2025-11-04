/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#error \
  "RMM requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined. Please add -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to the compiler flags (this is done automatically when using RMM via CMake)."
#endif  // LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>

namespace RMM_NAMESPACE {
namespace detail {
namespace polyfill {

template <class Resource>
inline constexpr bool resource = cuda::mr::synchronous_resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool resource_with = cuda::mr::synchronous_resource_with<Resource, Properties...>;
template <class Resource>
inline constexpr bool async_resource = cuda::mr::resource<Resource>;
template <class Resource, class... Properties>
inline constexpr bool async_resource_with = cuda::mr::resource_with<Resource, Properties...>;

}  // namespace polyfill
}  // namespace detail
}  // namespace RMM_NAMESPACE
