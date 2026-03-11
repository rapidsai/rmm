/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>

#include <cstddef>
#include <functional>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 */

/**
 * @brief Callback function type used by failure_callback_resource_adaptor
 *
 * The resource adaptor calls this function when a memory allocation throws a specified exception
 * type. The function decides whether the resource adaptor should try to allocate the memory again
 * or re-throw the exception.
 *
 * The callback function signature is:
 *     `bool failure_callback_t(std::size_t bytes, void* callback_arg)`
 *
 * The callback function is passed two parameters: `bytes` is the size of the failed memory
 * allocation and `arg` is the extra argument passed to the constructor of the
 * `failure_callback_resource_adaptor`. The callback function returns a Boolean where true means to
 * retry the memory allocation and false means to re-throw the exception.
 */
using failure_callback_t = std::function<bool(std::size_t, void*)>;

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
