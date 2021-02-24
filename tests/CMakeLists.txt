# =============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

include(CTest)

# Fetch GTest

CPMFindPackage(
  NAME GTest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.10.0
  VERSION 1.10.0
  GIT_SHALLOW TRUE
  OPTIONS "INSTALL_GTEST OFF"
  # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
  FIND_PACKAGE_ARGUMENTS "CONFIG")

if(GTest_ADDED)
  add_library(GTest::gtest ALIAS gtest)
  add_library(GTest::gmock ALIAS gmock)
  add_library(GTest::gtest_main ALIAS gtest_main)
  add_library(GTest::gmock_main ALIAS gmock_main)
endif()

# Auto-detect available GPU compute architectures

include(${RMM_SOURCE_DIR}/cmake/Modules/SetGPUArchs.cmake)
message(STATUS "RMM: Building tests with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

# Build options

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)

# compiler function

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTestInternal TEST_NAME TEST_SRC)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_include_directories(${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${RMM_SOURCE_DIR}>")
  target_link_libraries(${TEST_NAME} GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
                        pthread rmm)
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RMM_BINARY_DIR}/gtests>"
               CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(${TEST_NAME}
                             PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")
  target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
                                             -Wno-error=deprecated-declarations>)

  if(DISABLE_DEPRECATION_WARNING)
    target_compile_options(
      ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
    target_compile_options(${TEST_NAME}
                           PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
  endif(DISABLE_DEPRECATION_WARNING)

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction(ConfigureTestInternal)

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureTest TEST_NAME TEST_SRC)
  # Test with legacy default stream.
  ConfigureTestInternal("${TEST_NAME}" "${TEST_SRC}")

  # Test with per-thread default stream.
  string(REGEX REPLACE "_TEST$" "_PTDS_TEST" PTDS_TEST_NAME "${TEST_NAME}")
  ConfigureTestInternal("${PTDS_TEST_NAME}" "${TEST_SRC}")
  target_compile_definitions("${PTDS_TEST_NAME}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endfunction(ConfigureTest)

# test sources

# device mr tests

set(DEVICE_MR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_tests.cpp"
                       "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_multithreaded_tests.cpp")
ConfigureTest(DEVICE_MR_TEST "${DEVICE_MR_TEST_SRC}")

# pool mr tests

set(POOL_MR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/pool_mr_tests.cpp")
ConfigureTest(POOL_MR_TEST "${POOL_MR_TEST_SRC}")

# cuda_async mr tests
set(CUDA_ASYNC_MR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/cuda_async_mr_tests.cpp")
ConfigureTest(CUDA_ASYNC_MR_TEST "${CUDA_ASYNC_MR_TEST_SRC}")

# thrust allocator tests

set(THRUST_ALLOCATOR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/thrust_allocator_tests.cu")
ConfigureTest(THRUST_ALLOCATOR_TEST "${THRUST_ALLOCATOR_TEST_SRC}")

# polymorphic allocator tests

ConfigureTest(POLYMORPHIC_ALLOCATOR_TEST
              "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/polymorphic_allocator_tests.cpp")

# stream allocator adaptor tests

ConfigureTest(STREAM_ADAPTOR_TEST
              "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/stream_allocator_adaptor_tests.cpp")

# tracking adaptor tests

set(TRACKING_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/tracking_mr_tests.cpp")

ConfigureTest(TRACKING_TEST "${TRACKING_TEST_SRC}")

# limiting adaptor tests

set(LIMITING_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/limiting_mr_tests.cpp")
ConfigureTest(LIMITING_TEST "${LIMITING_TEST_SRC}")

# host mr tests

set(HOST_MR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mr/host/mr_tests.cpp")
ConfigureTest(HOST_MR_TEST "${HOST_MR_TEST_SRC}")

# cuda stream tests

set(CUDA_STREAM_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_tests.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_pool_tests.cpp")
ConfigureTest(CUDA_STREAM_TEST "${CUDA_STREAM_TEST_SRC}")

# device buffer tests

set(BUFFER_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device_buffer_tests.cu")
ConfigureTest(DEVICE_BUFFER_TEST "${BUFFER_TEST_SRC}")

# device scalar tests

set(SCALAR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device_scalar_tests.cpp")
ConfigureTest(DEVICE_SCALAR_TEST "${SCALAR_TEST_SRC}")

# logger tests

set(LOGGER_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/logger_tests.cpp")
ConfigureTest(LOGGER_TEST "${LOGGER_TEST_SRC}")

# uvector tests

set(DEVICE_UVECTOR_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector_tests.cpp")
ConfigureTest(DEVICE_UVECTOR_TEST "${DEVICE_UVECTOR_TEST_SRC}")
