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

#include <rmm/logger.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rmm {
/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * RMM_EXPECTS macro.
 *
 */
struct logic_error : public std::logic_error {
  using std::logic_error::logic_error;
};

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Exception thrown when an RMM allocation fails
 *
 */
class bad_alloc : public std::bad_alloc {
 public:
  bad_alloc(const char* w) : std::bad_alloc{}, _what{std::string{std::bad_alloc::what()} + ": " + w}
  {
  }

  bad_alloc(std::string const& w) : bad_alloc(w.c_str()) {}

  virtual ~bad_alloc() = default;

  virtual const char* what() const noexcept { return _what.c_str(); }

 private:
  std::string _what;
};

/**
 * @brief Exception thrown when attempting to access outside of a defined range
 *
 */
class out_of_range : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

}  // namespace rmm

#define STRINGIFY_DETAIL(x) #x
#define RMM_STRINGIFY(x)    STRINGIFY_DETAIL(x)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing `rmm::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws rmm::logic_error
 * RMM_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * RMM_EXPECTS(p != nullptr, std::runtime_error, "Unexpected nullptr");
 * ```
 * @param[in] _condition Expression that evaluates to true or false
 * @param[in] _expection_type The exception type to throw; must inherit
 *     `std::exception`. If not specified (i.e. if only two macro
 *     arguments are provided), defaults to `cudf::logic_error`
 * @param[in] _what  String literal description of why the exception was
 *     thrown, i.e. why `_condition` was expected to be true.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define RMM_EXPECTS(...)                                           \
  GET_RMM_EXPECTS_MACRO(__VA_ARGS__, RMM_EXPECTS_3, RMM_EXPECTS_2) \
  (__VA_ARGS__)
#define GET_RMM_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME
#define RMM_EXPECTS_3(_condition, _exception_type, _reason)              \
  (!!(_condition)) ? static_cast<void>(0) : throw _exception_type        \
  {                                                                      \
    "RMM failure at: " __FILE__ ":" RMM_STRINGIFY(__LINE__) ": " _reason \
  }
#define RMM_EXPECTS_2(_condition, _reason) RMM_EXPECTS_3(_condition, rmm::logic_error, _reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws `rmm::logic_error`
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
#define RMM_FAIL_2(_what, _exception_type) \
  throw _exception_type{"RMM failure at:" __FILE__ ":" RMM_STRINGIFY(__LINE__) ": " _what};
#define RMM_FAIL_1(_what) RMM_FAIL_2(_what, rmm::logic_error)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing `rmm::cuda_error`, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws `rmm::cuda_error` if `cudaMalloc` fails
 * RMM_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws `std::runtime_error` if `cudaMalloc` fails
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
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                \
                            RMM_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                            cudaGetErrorString(error)};                                      \
    }                                                                                        \
  } while (0)
#define RMM_CUDA_TRY_1(_call) RMM_CUDA_TRY_2(_call, rmm::cuda_error)

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
    assert(status__ == cudaSuccess);                                            \
  } while (0)
#endif

/**
 * @brief Assertion that logs a CRITICAL log message on failure.
 */
#ifdef NDEBUG
#define RMM_LOGGING_ASSERT(_expr) (void)0
#elif SPDLOG_ACTIVE_LEVEL < SPDLOG_LEVEL_OFF
#define RMM_LOGGING_ASSERT(_expr)                                                                 \
  do {                                                                                            \
    bool const success = (_expr);                                                                 \
    if (!success) {                                                                               \
      RMM_LOG_CRITICAL(                                                                           \
        "[" __FILE__ ":" RMM_STRINGIFY(__LINE__) "] Assertion " RMM_STRINGIFY(_expr) " failed."); \
      rmm::logger().flush();                                                                      \
      assert(success);                                                                            \
    }                                                                                             \
  } while (0)
#else
#define RMM_LOGGING_ASSERT(_expr) assert((_expr));
#endif
