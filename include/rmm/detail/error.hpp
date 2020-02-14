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

#include <cuda_runtime_api.h>
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
  logic_error(char const* const message) : std::logic_error(message) {}

  logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  cuda_error(std::string const& message) : std::runtime_error(message) {}
};
}  // namespace rmm

#define STRINGIFY_DETAIL(x) #x
#define RMM_STRINGIFY(x) STRINGIFY_DETAIL(x)

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
#define RMM_EXPECTS_3(_condition, _exception_type, _what) \
  (!!(_condition))                                        \
      ? static_cast<void>(0)                              \
      : throw _exception_type("RMM failure at: " __FILE__ \
                              ":" RMM_STRINGIFY(__LINE__) ": " _what)
#define RMM_EXPECTS_2(_condition, _reason) \
  RMM_EXPECTS_3(_condition, rmm::logic_error, _reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @throws `rmm::logic_error` always
 *
 * Example usage:
 * ```
 * RMM_FAIL("Unsupported code path");
 * ```
 *
 * @param[in] reason String literal description of the reason
 */
#define RMM_FAIL(reason)                             \
  throw rmm::logic_error("RMM failure at: " __FILE__ \
                         ":" RMM_STRINGIFY(__LINE__) ": " reason)

namespace rmm {
namespace detail {
inline void throw_cuda_error(cudaError_t error, const char* file,
                             unsigned int line) {
  throw rmm::cuda_error(
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}
}  // namespace detail
}  // namespace rmm

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define CUDA_TRY(call)                                           \
  do {                                                           \
    cudaError_t const status = (call);                           \
    if (cudaSuccess != status) {                                 \
      cudaGetLastError();                                        \
      rmm::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                            \
  } while (0);

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 *
 */
#ifndef NDEBUG
#define CHECK_CUDA(stream) CUDA_TRY(cudaStreamSynchronize(stream));
#else
#define CHECK_CUDA(stream) CUDA_TRY(cudaPeekAtLastError());
#endif