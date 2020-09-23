#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RMM_TESTS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

###################################################################################################
# - compiler function

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
  add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
  set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${RMM_SOURCE_DIR}>")

  target_link_libraries(${CMAKE_TEST_NAME} GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
                        pthread rmm)
  
  set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${RMM_BINARY_DIR}/gtests")

  target_compile_definitions(${CMAKE_TEST_NAME} PUBLIC
                        "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")    

  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - device mr tests

set(DEVICE_MR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_multithreaded_tests.cpp")
    
ConfigureTest(DEVICE_MR_TEST "${DEVICE_MR_TEST_SRC}")

###################################################################################################
# - device mr per-thread default stream tests

set(DEVICE_MR_PTDS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_multithreaded_tests.cpp")

ConfigureTest(DEVICE_MR_PTDS_TEST "${DEVICE_MR_PTDS_TEST_SRC}")
target_compile_definitions(DEVICE_MR_PTDS_TEST PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - pool mr tests

set(POOL_MR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/pool_mr_tests.cpp")
    
ConfigureTest(POOL_MR_TEST "${POOL_MR_TEST_SRC}")

set(POOL_MR_PTDS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/pool_mr_tests.cpp")

ConfigureTest(POOL_MR_PTDS_TEST "${POOL_MR_PTDS_TEST_SRC}")
target_compile_definitions(POOL_MR_PTDS_TEST PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - thrust allocator tests

set(THRUST_ALLOCATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/thrust_allocator_tests.cu")
    
ConfigureTest(THRUST_ALLOCATOR_TEST "${THRUST_ALLOCATOR_TEST_SRC}")

###################################################################################################
# - limiting adaptor tests

set(LIMITING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/limiting_mr_tests.cpp")

ConfigureTest(LIMITING_TEST "${LIMITING_TEST_SRC}")
target_compile_definitions(LIMITING_TEST PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - tracking adaptor tests

set(TRACKING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/tracking_mr_tests.cpp")

ConfigureTest(TRACKING_TEST "${TRACKING_TEST_SRC}")
target_compile_definitions(TRACKING_TEST PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - host mr tests

set(HOST_MR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/host/mr_tests.cpp")
    
ConfigureTest(HOST_MR_TEST "${HOST_MR_TEST_SRC}") 

###################################################################################################
# - device buffer tests

set(BUFFER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_buffer_tests.cu")

ConfigureTest(DEVICE_BUFFER_TEST "${BUFFER_TEST_SRC}")

###################################################################################################
# - device scalar tests

set(SCALAR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_scalar_tests.cpp")

ConfigureTest(DEVICE_SCALAR_TEST "${SCALAR_TEST_SRC}")

###################################################################################################
# - logger tests

set(LOGGER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/logger_tests.cpp")

ConfigureTest(LOGGER_TEST "${LOGGER_TEST_SRC}")

###################################################################################################
# - uvector tests

set(DEVICE_UVECTOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector_tests.cpp")

ConfigureTest(DEVICE_UVECTOR_TEST "${DEVICE_UVECTOR_TEST_SRC}")

###################################################################################################


enable_testing()
