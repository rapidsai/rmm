/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/std/type_traits>

namespace RMM_NAMESPACE {
namespace mr {

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
  rmm::detail::polyfill::resource<Resource>;

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
