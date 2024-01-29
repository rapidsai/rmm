/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <stdexcept>
#include <string>

namespace rmm {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * @ingroup errors
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
 * @ingroup errors
 *
 */
struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Exception thrown when an RMM allocation fails
 *
 * @ingroup errors
 *
 */
class bad_alloc : public std::bad_alloc {
 public:
  /**
   * @brief Constructs a bad_alloc with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  bad_alloc(const char* msg) : _what{std::string{std::bad_alloc::what()} + ": " + msg} {}

  /**
   * @brief Constructs a bad_alloc with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}

  /**
   * @briefreturn{The explanatory string}
   */
  [[nodiscard]] const char* what() const noexcept override { return _what.c_str(); }

 private:
  std::string _what;
};

/**
 * @brief Exception thrown when RMM runs out of memory
 *
 * @ingroup errors
 *
 * This error should only be thrown when we know for sure a resource is out of memory.
 */
class out_of_memory : public bad_alloc {
 public:
  /**
   * @brief Constructs an out_of_memory with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  out_of_memory(const char* msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}

  /**
   * @brief Constructs an out_of_memory with the error message.
   *
   * @param msg Message to be associated with the exception
   */
  out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}
};

/**
 * @brief Exception thrown when attempting to access outside of a defined range
 *
 * @ingroup errors
 *
 */
class out_of_range : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

}  // namespace rmm
