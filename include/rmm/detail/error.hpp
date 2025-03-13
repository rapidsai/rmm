/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>

#define STRINGIFY_DETAIL(x) #x
#define RMM_STRINGIFY(x)    STRINGIFY_DETAIL(x)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing rmm::logic_error, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws rmm::logic_error
 * RMM_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * RMM_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `rmm::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define RMM_EXPECTS(...)                                           \
  GET_RMM_EXPECTS_MACRO(__VA_ARGS__, RMM_EXPECTS_3, RMM_EXPECTS_2) \
  (__VA_ARGS__)
#define GET_RMM_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME
#define RMM_EXPECTS_3(_condition, _reason, _exception_type)                                 \
  do {                                                                                      \
    static_assert(std::is_base_of_v<std::exception, _exception_type>);                      \
  /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                            \
    (!!(_condition)) ? static_cast<void>(0)                                                 \
                 : throw _exception_type{std::string{"RMM failure at: "} + __FILE__ + ":" + \
                                         RMM_STRINGIFY(__LINE__) + ": " + _reason};         \
  } while (0)
#define RMM_EXPECTS_2(_condition, _reason) RMM_EXPECTS_3(_condition, _reason, rmm::logic_error)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws rmm::logic_error
 * RMM_FAIL("Unsupported code path");
 *
 * // Throws `std::runtime_error`
 * RMM_FAIL("Unsupported code path", std::runtime_error);
 * ```
 */
#define RMM_FAIL(...)                                     \
  GET_RMM_FAIL_MACRO(__VA_ARGS__, RMM_FAIL_2, RMM_FAIL_1) \
  (__VA_ARGS__)
#define GET_RMM_FAIL_MACRO(_1, _2, NAME, ...) NAME
#define RMM_FAIL_2(_what, _exception_type)                                                   \
  /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                             \
  throw _exception_type                                                                      \
  {                                                                                          \
    std::string{"RMM failure at:"} + __FILE__ + ":" + RMM_STRINGIFY(__LINE__) + ": " + _what \
  }
#define RMM_FAIL_1(_what) RMM_FAIL_2(_what, rmm::logic_error)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing rmm::cuda_error, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws rmm::cuda_error if `cudaMalloc` fails
 * RMM_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws std::runtime_error if `cudaMalloc` fails
 * RMM_CUDA_TRY(cudaMalloc(&p, 100), std::runtime_error);
 * ```
 *
 */
#define RMM_CUDA_TRY(...)                                             \
  GET_RMM_CUDA_TRY_MACRO(__VA_ARGS__, RMM_CUDA_TRY_2, RMM_CUDA_TRY_1) \
  (__VA_ARGS__)
#define GET_RMM_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define RMM_CUDA_TRY_2(_call, _exception_type)                                               \
  do {                                                                                       \
    cudaError_t const error = (_call);                                                       \
    if (cudaSuccess != error) {                                                              \
      cudaGetLastError();                                                                    \
      /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                         \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                \
                            RMM_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                            cudaGetErrorString(error)};                                      \
    }                                                                                        \
  } while (0)
#define RMM_CUDA_TRY_1(_call) RMM_CUDA_TRY_2(_call, rmm::cuda_error)

/**
 * @brief Error checking macro for CUDA memory allocation calls.
 *
 * Invokes a CUDA memory allocation function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing rmm::bad_alloc, but when `cudaErrorMemoryAllocation` is returned,
 * rmm::out_of_memory is thrown instead.
 *
 * Can be called with either 1 or 2 arguments:
 * - RMM_CUDA_TRY_ALLOC(cuda_call): Performs error checking without specifying bytes
 * - RMM_CUDA_TRY_ALLOC(cuda_call, num_bytes): Includes the number of bytes in the error message
 */
#define RMM_CUDA_TRY_ALLOC(...)                                                         \
  GET_RMM_CUDA_TRY_ALLOC_MACRO(__VA_ARGS__, RMM_CUDA_TRY_ALLOC_2, RMM_CUDA_TRY_ALLOC_1) \
  (__VA_ARGS__)
#define GET_RMM_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME

#define RMM_CUDA_TRY_ALLOC_2(_call, num_bytes)                                          \
  do {                                                                                  \
    cudaError_t const error = (_call);                                                  \
    if (cudaSuccess != error) {                                                         \
      cudaGetLastError();                                                               \
      auto const msg = std::string{"CUDA error (failed to allocate "} +                 \
                       std::to_string(num_bytes) + " bytes) at: " + __FILE__ + ":" +    \
                       RMM_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                       cudaGetErrorString(error);                                       \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }        \
      throw rmm::bad_alloc{msg};                                                        \
    }                                                                                   \
  } while (0)

#define RMM_CUDA_TRY_ALLOC_1(_call)                                                                \
  do {                                                                                             \
    cudaError_t const error = (_call);                                                             \
    if (cudaSuccess != error) {                                                                    \
      cudaGetLastError();                                                                          \
      auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" + RMM_STRINGIFY(__LINE__) + \
                       ": " + cudaGetErrorName(error) + " " + cudaGetErrorString(error);           \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }                   \
      throw rmm::bad_alloc{msg};                                                                   \
    }                                                                                              \
  } while (0)

/**
 * @brief Error checking macro similar to `assert` for CUDA runtime API calls
 *
 * This utility should be used in situations where extra error checking is desired in "Debug"
 * builds, or in situations where an error case cannot throw an exception (such as a class
 * destructor).
 *
 * In "Release" builds, simply invokes the `_call`.
 *
 * In "Debug" builds, invokes `_call` and uses `assert` to verify the returned `cudaError_t` is
 * equal to `cudaSuccess`.
 *
 *
 * Replaces usecases such as:
 * ```
 * auto status = cudaRuntimeApi(...);
 * assert(status == cudaSuccess);
 * ```
 *
 * Example:
 * ```
 * RMM_ASSERT_CUDA_SUCCESS(cudaRuntimeApi(...));
 * ```
 *
 */
#ifdef NDEBUG
#define RMM_ASSERT_CUDA_SUCCESS(_call) \
  do {                                 \
    (_call);                           \
  } while (0);
#else
#define RMM_ASSERT_CUDA_SUCCESS(_call)                                          \
  do {                                                                          \
    cudaError_t const status__ = (_call);                                       \
    if (status__ != cudaSuccess) {                                              \
      std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " " \
                << cudaGetErrorString(status__) << std::endl;                   \
    }                                                                           \
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */   \
    assert(status__ == cudaSuccess);                                            \
  } while (0)
#endif
