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
#define EXCEPTION_HPP

#include <exception>

/** ---------------------------------------------------------------------------*
 * @file exceptions.hpp
 * @brief Custom exceptions used by RMM.
 *
 * Exceptions for errors occuring due to out-of-memory, CUDA, and CNMEM errors.
 * 
 * --------------------------------------------------------------------------**/
namespace rmm{
struct bad_alloc: public std::bad_alloc
{
  bad_alloc(const char* msg_, const char* file_, unsigned int line_) : msg{msg_}, file{file_}, line {line_}
  { }

  bad_alloc(const char* file_, unsigned int line_) : file{file_}, line {line_}
  { }

  bad_alloc(const char* msg_) : msg{msg_}{}

  bad_alloc(){} const char * what () const noexcept
  {
    std::string message{"RMM out of memory."};

    if(not msg.empty())
      message += msg;
    
    if(not file.empty())
      message += " File: " + file + " line: " + std::to_string(line);

    return message.c_str();
  }

private:
  std::string const msg;
  std::string const file;
  unsigned int line{};
};

struct cuda_error: public std::runtime_error
{
  cuda_error(const char* msg_, const char* file_, unsigned int line_, cudaError_t err_) 
    : msg{msg_}, file{file_}, line{line_}, error{err_} {}

  cuda_error(const char* file_, unsigned int line_, cudaError_t err_) 
    : file{file_}, line{line_}, error{err_} {}

  cuda_error(cudaError_t err_) : error{err_}{}

  const char * what () const noexcept
  {
    std::string message{"RMM CUDA error."};

    if(not msg.empty())
      message += msg;
    
    if(not file.empty())
      message += " File: " + file + " line: " + std::to_string(line);

    if(error != cudaSuccess)
      message += " error code: " + std::to_string(error);

    return message.c_str();
  }

private:
  std::string const msg;
  std::string const file;
  unsigned int line{};
  cudaError_t error{cudaSuccess};
};

struct cnmem_error: public std::runtime_error
{
  cuda_error(const char* msg_, const char* file_, unsigned int line_, cnmemStatus_t err_) 
    : msg{msg_}, file{file_}, line{line_}, error{err_} {}

  cuda_error(const char* file_, unsigned int line_, cnmemStatus_t err_) 
    : file{file_}, line{line_}, error{err_} {}

  cuda_error(cnmemStatus_t err_) : error{err_}{}

  const char * what () const noexcept
  {
    std::string message{"RMM CNMEM error."};

    if(not msg.empty())
      message += msg;
    
    if(not file.empty())
      message += " File: " + file + " line: " + std::to_string(line);

    if(0 != error)
      message += " error code: " + std::to_string(error);

    return message.c_str();
  }

private:
  std::string const msg;
  std::string const file;
  unsigned int line{};
  cnmemStatus_t error{0};
};
} // namespace rmm
#endif

