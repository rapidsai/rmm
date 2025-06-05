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

#include <rmm/detail/error.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

/*
 * This file contains tests for the RMM error macros.
 *
 * RMM macros are not public API but are used externally, so we test them to
 * avoid regressions.
 *
 * The macros are tested for:
 * - Successful operations (should not throw)
 * - Failed operations (should throw the appropriate exception)
 * - Error message formatting
 * - Actual CUDA operations
 */

// Test RMM_EXPECTS macro with condition that evaluates to true (should not throw)
TEST(ErrorMacrosTest, ExpectsNoThrow)
{
  EXPECT_NO_THROW(RMM_EXPECTS(true, "This should not throw"));
  EXPECT_NO_THROW(RMM_EXPECTS(true, "This should not throw", std::runtime_error));
}

// Test RMM_EXPECTS macro with condition that evaluates to false (should throw)
TEST(ErrorMacrosTest, ExpectsThrow)
{
  EXPECT_THROW(RMM_EXPECTS(false, "Expected exception"), rmm::logic_error);
  EXPECT_THROW(RMM_EXPECTS(false, "Expected runtime error", std::runtime_error),
               std::runtime_error);
}

// Test RMM_FAIL macro (should always throw)
TEST(ErrorMacrosTest, FailThrow)
{
  EXPECT_THROW(RMM_FAIL("This should throw logic_error"), rmm::logic_error);
  EXPECT_THROW(RMM_FAIL("This should throw runtime_error", std::runtime_error), std::runtime_error);
}

// Test RMM_CUDA_TRY macro with successful CUDA call (should not throw)
TEST(ErrorMacrosTest, CudaTryNoThrow)
{
  EXPECT_NO_THROW(RMM_CUDA_TRY(cudaSuccess));
  EXPECT_NO_THROW(RMM_CUDA_TRY(cudaSuccess, std::runtime_error));
}

// Test RMM_CUDA_TRY macro with failed CUDA call (should throw)
TEST(ErrorMacrosTest, CudaTryThrow)
{
  EXPECT_THROW(RMM_CUDA_TRY(cudaErrorInvalidValue), rmm::cuda_error);
  EXPECT_THROW(RMM_CUDA_TRY(cudaErrorInvalidValue, std::runtime_error), std::runtime_error);
}

// Test RMM_CUDA_TRY_ALLOC macro with successful CUDA call (should not throw)
TEST(ErrorMacrosTest, CudaTryAllocNoThrow)
{
  EXPECT_NO_THROW(RMM_CUDA_TRY_ALLOC(cudaSuccess));
  EXPECT_NO_THROW(RMM_CUDA_TRY_ALLOC(cudaSuccess, 1024));
}

// Test RMM_CUDA_TRY_ALLOC macro with general CUDA error (should throw bad_alloc)
TEST(ErrorMacrosTest, CudaTryAllocThrowBadAlloc)
{
  EXPECT_THROW(RMM_CUDA_TRY_ALLOC(cudaErrorInvalidValue), rmm::bad_alloc);
  EXPECT_THROW(RMM_CUDA_TRY_ALLOC(cudaErrorInvalidValue, 1024), rmm::bad_alloc);
}

// Test RMM_CUDA_TRY_ALLOC macro with memory allocation error (should throw out_of_memory)
TEST(ErrorMacrosTest, CudaTryAllocThrowOutOfMemory)
{
  EXPECT_THROW(RMM_CUDA_TRY_ALLOC(cudaErrorMemoryAllocation), rmm::out_of_memory);
  EXPECT_THROW(RMM_CUDA_TRY_ALLOC(cudaErrorMemoryAllocation, 1024), rmm::out_of_memory);
}

// Test RMM_ASSERT_CUDA_SUCCESS macro
// Note: This test is limited since the macro behavior changes based on NDEBUG
// In release builds (NDEBUG defined), it just executes the call
// In debug builds, it asserts on failure which can't be caught in tests
TEST(ErrorMacrosTest, AssertCudaSuccess)
{
  // This should always work regardless of build type
  EXPECT_NO_FATAL_FAILURE(RMM_ASSERT_CUDA_SUCCESS([]() { return cudaSuccess; }()));

// We can't really test the failure case properly in a unit test
// since it would call assert() which terminates the program
#ifdef NDEBUG
  // In release builds, this should not crash
  EXPECT_NO_FATAL_FAILURE(RMM_ASSERT_CUDA_SUCCESS([]() { return cudaErrorInvalidValue; }()));
#endif
}

// Test that error messages contain expected information
TEST(ErrorMacrosTest, ErrorMessages)
{
  // Test RMM_EXPECTS error message
  try {
    RMM_EXPECTS(false, "Test message");
    FAIL() << "Expected RMM_EXPECTS to throw an exception";
  } catch (const rmm::logic_error& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("RMM failure at:") != std::string::npos);
    EXPECT_TRUE(error_message.find("Test message") != std::string::npos);
  }

  // Test RMM_FAIL error message
  try {
    RMM_FAIL("Test failure message");
    FAIL() << "Expected RMM_FAIL to throw an exception";
  } catch (const rmm::logic_error& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("RMM failure at:") != std::string::npos);
    EXPECT_TRUE(error_message.find("Test failure message") != std::string::npos);
  }

  // Test RMM_CUDA_TRY error message
  try {
    RMM_CUDA_TRY(cudaErrorInvalidValue);
    FAIL() << "Expected RMM_CUDA_TRY to throw an exception";
  } catch (const rmm::cuda_error& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("CUDA error at:") != std::string::npos);
    EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
  }

  // Test RMM_CUDA_TRY_ALLOC error message (without bytes)
  try {
    RMM_CUDA_TRY_ALLOC(cudaErrorInvalidValue);
    FAIL() << "Expected RMM_CUDA_TRY_ALLOC to throw an exception";
  } catch (const rmm::bad_alloc& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("CUDA error at:") != std::string::npos);
    EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
  }

  // Test RMM_CUDA_TRY_ALLOC error message (with bytes)
  try {
    RMM_CUDA_TRY_ALLOC(cudaErrorInvalidValue, 1024);
    FAIL() << "Expected RMM_CUDA_TRY_ALLOC to throw an exception";
  } catch (const rmm::bad_alloc& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("CUDA error (failed to allocate 1024 bytes)") !=
                std::string::npos);
    EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
  }

  // Test RMM_CUDA_TRY_ALLOC out_of_memory error message
  try {
    RMM_CUDA_TRY_ALLOC(cudaErrorMemoryAllocation, 2048);
    FAIL() << "Expected RMM_CUDA_TRY_ALLOC to throw an exception";
  } catch (const rmm::out_of_memory& e) {
    std::string error_message = e.what();
    EXPECT_TRUE(error_message.find("out_of_memory") != std::string::npos);
    EXPECT_TRUE(error_message.find("failed to allocate 2048 bytes") != std::string::npos);
    EXPECT_TRUE(error_message.find("out of memory") != std::string::npos);
  }
}

// Test actual CUDA operations with the macros
TEST(ErrorMacrosTest, ActualCudaOperations)
{
  // Test successful memory allocation and free
  void* d_ptr                           = nullptr;
  constexpr size_t test_allocation_size = 1024;

  EXPECT_NO_THROW(
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&d_ptr, test_allocation_size), test_allocation_size));
  ASSERT_NE(d_ptr, nullptr);

  EXPECT_NO_THROW(RMM_CUDA_TRY(cudaFree(d_ptr)));

  // Test successful CUDA operation
  std::array<int, 5> h_data = {1, 2, 3, 4, 5};
  int* d_data               = nullptr;

  EXPECT_NO_THROW(RMM_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(h_data)),
                                     sizeof(h_data)));

  ASSERT_NE(d_data, nullptr);

  EXPECT_NO_THROW(
    RMM_CUDA_TRY(cudaMemcpy(d_data, h_data.data(), sizeof(h_data), cudaMemcpyHostToDevice)));

  std::array<int, 5> h_result = {0};
  EXPECT_NO_THROW(
    RMM_CUDA_TRY(cudaMemcpy(h_result.data(), d_data, sizeof(h_result), cudaMemcpyDeviceToHost)));

  for (size_t i = 0; i < h_data.size(); ++i) {
    EXPECT_EQ(h_data[i], h_result[i]);
  }

  EXPECT_NO_THROW(RMM_CUDA_TRY(cudaFree(d_data)));
}
