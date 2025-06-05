# =============================================================================
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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
option(CODE_COVERAGE "Enable generating code coverage with gcov." OFF)

include(rapids-test)
rapids_test_init()

# Ensure tests are using the conda env, so they have the correct release/debug compile flags
rapids_cmake_support_conda_env(conda_env)

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTestInternal TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
  target_include_directories(${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${RMM_SOURCE_DIR}>")
  target_link_libraries(
    ${TEST_NAME} PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main pthread rmm
                         $<TARGET_NAME_IF_EXISTS:conda_env>)
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RMM_BINARY_DIR}/gtests>"
               CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(
    ${TEST_NAME} PUBLIC "RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${RMM_LOGGING_LEVEL}")
  target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror>)

  if(DISABLE_DEPRECATION_WARNING)
    target_compile_options(
      ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
    target_compile_options(${TEST_NAME}
                           PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
  endif()

  if(CODE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

      set(KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp)
      make_directory(${KEEP_DIR})
      target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep
                                                 --keep-dir=${KEEP_DIR}>)
      target_compile_options(
        ${TEST_NAME}
        PUBLIC
          $<$<COMPILE_LANGUAGE:CUDA>:-O0
          -Xcompiler=--coverage,-fprofile-abs-path,-fkeep-inline-functions,-fno-elide-constructors>)
      target_compile_options(
        ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O0 --coverage -fprofile-abs-path
                            -fkeep-inline-functions -fno-elide-constructors>)
      target_link_options(${TEST_NAME} PRIVATE --coverage)
      target_link_libraries(${TEST_NAME} PRIVATE gcov)
    endif()

    # Add coverage-generated files to clean target
    list(APPEND COVERAGE_CLEAN_FILES "**/*.gcno" "**/*.gcda")
    set_property(
      TARGET ${TEST_NAME}
      APPEND
      PROPERTY ADDITIONAL_CLEAN_FILES ${COVERAGE_CLEAN_FILES})
  endif()

endfunction()

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureTest TEST_NAME)

  set(options LINK_DRIVER)
  set(one_value CUDART GPUS PERCENT)
  set(multi_value)
  cmake_parse_arguments(_RMM_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _RMM_TEST_GPUS AND NOT DEFINED _RMM_TEST_PERCENT)
    set(_RMM_TEST_GPUS 1)
    set(_RMM_TEST_PERCENT 5)
  endif()
  if(NOT DEFINED _RMM_TEST_GPUS)
    set(_RMM_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _RMM_TEST_PERCENT)
    set(_RMM_TEST_PERCENT 100)
  endif()

  if(_RMM_TEST_CUDART STREQUAL SHARED)
    set(cudart_link_libs $<COMPILE_ONLY:rmm> CUDA::cudart)
  elseif(_RMM_TEST_CUDART STREQUAL STATIC)
    set(cudart_link_libs $<COMPILE_ONLY:rmm> CUDA::cudart_static)
  else()
    set(cudart_link_libs rmm)
  endif()

  # Test with legacy default stream.
  ConfigureTestInternal(${TEST_NAME} ${_RMM_TEST_UNPARSED_ARGUMENTS})
  target_link_libraries(${TEST_NAME} PRIVATE ${cudart_link_libs})
  if(_RMM_TEST_LINK_DRIVER)
    target_link_libraries(${TEST_NAME} PRIVATE CUDA::cuda_driver)
  endif()
  # Test with per-thread default stream.
  string(REGEX REPLACE "_TEST$" "_PTDS_TEST" PTDS_TEST_NAME "${TEST_NAME}")
  ConfigureTestInternal("${PTDS_TEST_NAME}" ${_RMM_TEST_UNPARSED_ARGUMENTS})
  target_compile_definitions("${PTDS_TEST_NAME}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
  target_link_libraries(${PTDS_TEST_NAME} PRIVATE ${cudart_link_libs})
  if(_RMM_TEST_LINK_DRIVER)
    target_link_libraries(${PTDS_TEST_NAME} PRIVATE CUDA::cuda_driver)
  endif()

  foreach(name ${TEST_NAME} ${PTDS_TEST_NAME} ${NS_TEST_NAME})
    rapids_test_add(
      NAME ${name}
      COMMAND ${TEST_NAME}
      GPUS ${_RMM_TEST_GPUS}
      PERCENT ${_RMM_TEST_PERCENT}
      INSTALL_COMPONENT_SET testing)
  endforeach()

endfunction()

# test sources

# device mr_ref tests
ConfigureTest(DEVICE_MR_REF_TEST mr/device/mr_ref_tests.cpp
              mr/device/mr_ref_multithreaded_tests.cpp GPUS 1 PERCENT 100)

# general adaptor tests
ConfigureTest(ADAPTOR_TEST mr/device/adaptor_tests.cpp)

# pool mr tests
ConfigureTest(POOL_MR_TEST mr/device/pool_mr_tests.cpp GPUS 1 PERCENT 100)

ConfigureTest(HWDECOMPRESS_TEST mr/device/hwdecompress_tests.cpp LINK_DRIVER)

# cuda_async mr tests
ConfigureTest(CUDA_ASYNC_MR_STATIC_CUDART_TEST mr/device/cuda_async_mr_tests.cpp GPUS 1 PERCENT 60
              CUDART STATIC)
ConfigureTest(CUDA_ASYNC_MR_SHARED_CUDART_TEST mr/device/cuda_async_mr_tests.cpp GPUS 1 PERCENT 60
              CUDART SHARED)

# thrust allocator tests
ConfigureTest(THRUST_ALLOCATOR_TEST mr/device/thrust_allocator_tests.cu GPUS 1 PERCENT 60)

# polymorphic allocator tests
ConfigureTest(POLYMORPHIC_ALLOCATOR_TEST mr/device/polymorphic_allocator_tests.cpp)

# stream allocator adaptor tests
ConfigureTest(STREAM_ADAPTOR_TEST mr/device/stream_allocator_adaptor_tests.cpp)

# statistics adaptor tests
ConfigureTest(STATISTICS_TEST mr/device/statistics_mr_tests.cpp)

# tracking adaptor tests
ConfigureTest(TRACKING_TEST mr/device/tracking_mr_tests.cpp)

# out-of-memory callback adaptor tests
ConfigureTest(FAILURE_CALLBACK_TEST mr/device/failure_callback_mr_tests.cpp)

# prefetch adaptor tests
ConfigureTest(PREFETCH_ADAPTOR_TEST mr/device/prefetch_resource_adaptor_tests.cpp)

# aligned adaptor tests
ConfigureTest(ALIGNED_TEST mr/device/aligned_mr_tests.cpp)

# limiting adaptor tests
ConfigureTest(LIMITING_TEST mr/device/limiting_mr_tests.cpp)

# host mr_ref tests
ConfigureTest(HOST_MR_REF_TEST mr/host/mr_ref_tests.cpp)

# pinned pool mr tests
ConfigureTest(PINNED_POOL_MR_TEST mr/host/pinned_pool_mr_tests.cpp)

# cuda stream tests
ConfigureTest(CUDA_STREAM_TEST cuda_stream_tests.cpp cuda_stream_pool_tests.cpp)

# device buffer tests
ConfigureTest(DEVICE_BUFFER_TEST device_buffer_tests.cu)

# device scalar tests
ConfigureTest(DEVICE_SCALAR_TEST device_scalar_tests.cpp)

# uvector tests
ConfigureTest(DEVICE_UVECTOR_TEST device_uvector_tests.cpp GPUS 1 PERCENT 60)

# prefetch tests
ConfigureTest(PREFETCH_TEST prefetch_tests.cpp)

# logger tests
ConfigureTest(LOGGER_TEST logger_tests.cpp)

# arena MR tests
ConfigureTest(ARENA_MR_TEST mr/device/arena_mr_tests.cpp GPUS 1 PERCENT 100)

# binning MR tests
ConfigureTest(BINNING_MR_TEST mr/device/binning_mr_tests.cpp)

# callback memory resource tests
ConfigureTest(CALLBACK_MR_TEST mr/device/callback_mr_tests.cpp)

# system memory resource tests
ConfigureTest(SYSTEM_MR_TEST mr/device/system_mr_tests.cu GPUS 1 PERCENT 100)

# container multidevice tests
ConfigureTest(CONTAINER_MULTIDEVICE_TEST container_multidevice_tests.cu)

# error macros tests
ConfigureTest(ERROR_MACROS_TEST error_macros_tests.cpp)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/librmm)
