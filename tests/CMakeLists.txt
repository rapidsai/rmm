#=============================================================================
# Copyright (c) 2018, NVIDIA CORPORATION.
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
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cudart rmm
                            spdlog::spdlog_header_only)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${GTEST_INCLUDE_DIR}"
                    "${CUB_INCLUDE_DIR}"
                    "${THRUST_INCLUDE_DIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

# CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link
# directories for nvcc
link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" 
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - rmm tests -------------------------------------------------------------------------------------

###################################################################################################
###################################################################################################
# - device mr tests --------------------------------------------------------------------------------

set(DEVICE_MR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_multithreaded_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/thrust_allocator_tests.cu")
    

ConfigureTest(DEVICE_MR_TEST "${DEVICE_MR_TEST_SRC}")

###################################################################################################
###################################################################################################
# - device mr per-thread default stream tests------------------------------------------------------


set(DEVICE_MR_PTDS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/device/mr_multithreaded_tests.cpp")

ConfigureTest(DEVICE_MR_PTDS_TEST "${DEVICE_MR_PTDS_TEST_SRC}")
target_compile_definitions(DEVICE_MR_PTDS_TEST PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)


###################################################################################################
###################################################################################################
# - host mr tests ----------------------------------------------------------------------------------

set(HOST_MR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/mr/host/mr_tests.cpp")
    
ConfigureTest(HOST_MR_TEST "${HOST_MR_TEST_SRC}") 

###################################################################################################
###################################################################################################
# - device buffer tests ----------------------------------------------------------------------------

set(BUFFER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_buffer_tests.cu")

ConfigureTest(DEVICE_BUFFER_TEST "${BUFFER_TEST_SRC}")

###################################################################################################
###################################################################################################
# - device scalar tests ----------------------------------------------------------------------------

set(SCALAR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_scalar_tests.cpp")

ConfigureTest(DEVICE_SCALAR_TEST "${SCALAR_TEST_SRC}")
###################################################################################################
# - logger tests ----------------------------------------------------------------------------------

set(LOGGER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/logger_tests.cpp")

ConfigureTest(LOGGER_TEST "${LOGGER_TEST_SRC}")

###################################################################################################
# - uvector tests ---------------------------------------------------------------------------------

set(DEVICE_UVECTOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector_tests.cu")

ConfigureTest(DEVICE_UVECTOR_TEST "${DEVICE_UVECTOR_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
