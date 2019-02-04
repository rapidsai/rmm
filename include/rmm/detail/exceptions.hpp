/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <cuda_runtime_api.h>
#include <exception>
#include <limits>

/** ---------------------------------------------------------------------------*
 * @file exceptions.hpp
 * @brief Custom exceptions used by RMM.
 *
 * Exceptions for errors occuring due to out-of-memory, CUDA, and CNMEM errors.
 *
 * --------------------------------------------------------------------------**/
namespace rmm {

struct exception : public std::exception {
  exception(rmmError_t err_, const char* file_ = nullptr,
            unsigned int line_ = std::numeric_limits<unsigned int>::max())
      : file{file_}, line{line_}, error{err_} {
    if (not file.empty()) {
      msg += " File: " + file + " line: " + std::to_string(line);
    }
  }

  const char* what() const noexcept { return msg.c_str(); }

 private:
  std::string msg{"RMM Exception."};
  std::string const file{};
  unsigned int const line{};
  rmmError_t const error{};
};

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when an out of memory error occurs in RMM.
 *
 *---------------------------------------------------------------------------**/
struct bad_alloc : public std::bad_alloc {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new bad_alloc exception
   *
   * @param[in] file_ Optional filename where error occured (should be populated
   * by __FILE__ macro)
   * @param[in] line_ Optional line number where error occured (should be
   * populated by __LINE__ macro)
   *---------------------------------------------------------------------------**/
  bad_alloc(const char* file_, unsigned int line_) : file{file_}, line{line_} {
    if (not file.empty()) {
      msg += " File: " + file + " line: " + std::to_string(line);
    }
  }

  bad_alloc() = default;

  /**---------------------------------------------------------------------------*
   * @brief Returns explanatory string for this exception
   *
   *---------------------------------------------------------------------------**/
  const char* what() const noexcept { return msg.c_str(); }

 private:
  std::string msg{"RMM out of memory excpetion."};  ///< Explanatory string
  std::string const file{};   ///< File name where exception occured
  unsigned int const line{};  ///< Line number where exceptin occured
};

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::exception {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new cuda_error exception
   *
   * @param[in] err_ The CUDA error code that resulted from the unsuccesfull
   *CUDA function
   * @param[in] file_ Optional filename where error occured (should be populated
   * by __FILE__ macro)
   * @param[in] line_ Optional line number where error occured (should be
   * populated by __LINE__ macro)
   *---------------------------------------------------------------------------**/
  cuda_error(cudaError_t err_, const char* file_ = nullptr,
             unsigned int line_ = std::numeric_limits<unsigned int>::max())
      : file{file_}, line{line_}, error{err_} {
    if (not file.empty()) {
      msg += " File: " + file + " line: " + std::to_string(line);
    }

    msg += " error code: " + std::to_string(error) + " " +
           cudaGetErrorName(error) + " " + cudaGetErrorString(error);
  }

  cuda_error() = delete;

  /**---------------------------------------------------------------------------*
   * @brief Returns explanatory string for this exception
   *
   *---------------------------------------------------------------------------**/
  const char* what() const noexcept { return msg.c_str(); }

 private:
  std::string msg{"RMM CUDA exception."};  ///< Explanatory string
  std::string const file;     ///< File name where exception occured
  unsigned int const line{};  ///< Line numeber where exception occured
  cudaError_t const error;    ///< CUDA error code returned from the
                              ///< unsuccesfull CUDA function
};

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CNMEM error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cnmem_error : public std::exception {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new cnmem_error exception.
   *
   * @param[in] err_ The CNMEM error code that resulted from the unsuccesfull
   * CNMEM function.
   * @param[in] file_ Optional filename where error occured (should be populated
   * by __FILE__ macro)
   * @param[in] line_ Optional line number where error occured (should be
   * populated by __LINE__ macro)*
   *---------------------------------------------------------------------------**/
  cnmem_error(cnmemStatus_t err_, const char* file_ = nullptr,
              unsigned int line_ = std::numeric_limits<unsigned int>::max())
      : file{file_}, line{line_}, error{err_} {
    if (not file.empty()) {
      msg += " File: " + file + " line: " + std::to_string(line);
    }

    msg += " error code: " + std::to_string(error);
  }

  cnmem_error() = delete;

  /**---------------------------------------------------------------------------*
   * @brief Returns explanatory string for this exception
   *
   *---------------------------------------------------------------------------**/
  const char* what() const noexcept { return msg.c_str(); }

 private:
  std::string msg{"RMM CNMEM exception."};  ///< Explanatory string
  std::string const file;     ///< File name where exception occurred
  unsigned int const line{};  ///< Line number where exception occurred
  cnmemStatus_t const error;  ///< CNMEM error code returned from
                              ///< unsucessful CNMEM function
};
}  // namespace rmm

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper to check for error in RMM API calls.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK(call, file, line)                \
  do {                                             \
    rmmError_t error = (call);                     \
    if (error != RMM_SUCCESS) {                    \
      throw rmm::exception(error, (file), (line)); \
    }                                              \
  } while (0)

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for RMM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CUDA(call, file, line)                \
  do {                                                  \
    cudaError_t cudaError = (call);                     \
    if (cudaError == cudaErrorMemoryAllocation)         \
      throw rmm::bad_alloc((file), (line));             \
    else if (cudaError != cudaSuccess)                  \
      throw rmm::cuda_error(cudaError, (file), (line)); \
  } while (0)

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for CNMEM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CNMEM(call, file, line)        \
  do {                                           \
    cnmemStatus_t error = (call);                \
    if (CNMEM_STATUS_SUCCESS != error) {         \
      throw rmm::cnmem_error(error, file, line); \
    }                                            \
  } while (0)

#endif
