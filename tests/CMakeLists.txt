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

# Build options
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)

# compiler function

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTestInternal TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
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
  endif()

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureTest TEST_NAME)
  # Test with legacy default stream.
  ConfigureTestInternal(${TEST_NAME} ${ARGN})

  # Test with per-thread default stream.
  string(REGEX REPLACE "_TEST$" "_PTDS_TEST" PTDS_TEST_NAME "${TEST_NAME}")
  ConfigureTestInternal("${PTDS_TEST_NAME}" ${ARGN})
  target_compile_definitions("${PTDS_TEST_NAME}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endfunction()

# test sources

# device mr tests
ConfigureTest(DEVICE_MR_TEST mr/device/mr_tests.cpp mr/device/mr_multithreaded_tests.cpp)

# pool mr tests
ConfigureTest(POOL_MR_TEST mr/device/pool_mr_tests.cpp)

# cuda_async mr tests
ConfigureTest(CUDA_ASYNC_MR_TEST mr/device/cuda_async_mr_tests.cpp)

# thrust allocator tests
ConfigureTest(THRUST_ALLOCATOR_TEST mr/device/thrust_allocator_tests.cu)

# polymorphic allocator tests
ConfigureTest(POLYMORPHIC_ALLOCATOR_TEST mr/device/polymorphic_allocator_tests.cpp)

# stream allocator adaptor tests
ConfigureTest(STREAM_ADAPTOR_TEST mr/device/stream_allocator_adaptor_tests.cpp)

# statistics adaptor tests
ConfigureTest(STATISTICS_TEST mr/device/statistics_mr_tests.cpp)

# tracking adaptor tests
ConfigureTest(TRACKING_TEST mr/device/tracking_mr_tests.cpp)

# aligned adaptor tests
ConfigureTest(ALIGNED_TEST mr/device/aligned_mr_tests.cpp)

# limiting adaptor tests
ConfigureTest(LIMITING_TEST mr/device/limiting_mr_tests.cpp)

# host mr tests
ConfigureTest(HOST_MR_TEST mr/host/mr_tests.cpp)

# cuda stream tests
ConfigureTest(CUDA_STREAM_TEST cuda_stream_tests.cpp cuda_stream_pool_tests.cpp)

# device buffer tests
ConfigureTest(DEVICE_BUFFER_TEST device_buffer_tests.cu)

# device scalar tests
ConfigureTest(DEVICE_SCALAR_TEST device_scalar_tests.cpp)

# logger tests
ConfigureTest(LOGGER_TEST logger_tests.cpp)

# uvector tests
ConfigureTest(DEVICE_UVECTOR_TEST device_uvector_tests.cpp)
