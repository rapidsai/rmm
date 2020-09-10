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
cmake_minimum_required(VERSION 3.17...3.18 FATAL_ERROR)

project(RMM VERSION 0.16.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER MATCHES ".*clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "RMM: Enabling the GLIBCXX11 ABI")
    else()
        if(CMAKE_CXX_COMPILER MATCHES ".*clang")
            message(FATAL_ERROR "Using clang with CMAKE_CXX11_ABI=Off is untested")
        endif(CMAKE_CXX_COMPILER MATCHES ".*clang")
        message(STATUS "RMM: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER MATCHES ".*clang")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
if(DISABLE_DEPRECATION_WARNING)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNING)

# cudart can be statically linked or dynamically linked the python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)

option(USE_NVTX "Build with NVTX support" OFF)

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - third party dependencies-----------------------------------------------------------------------
# add third party dependencies using CMake FetchContent

add_subdirectory(thirdparty)

###################################################################################################
# - cnmem ---------------------------------------------------------------------------------
# Temporarily needed to have all RMM includes under a single directory
configure_file("${CNMEM_INCLUDE_DIR}/cnmem.h"
	       "${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/detail/cnmem.h" COPYONLY)

###################################################################################################
# - per-thread default stream option --------------------------------------------------------------
# This needs to be defined first so tests and benchmarks can inherit it.

if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif(PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)
  include(ConfigureGoogleBenchmark)

  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)
endif(BUILD_BENCHMARKS)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(rmm SHARED "${CNMEM_SOURCE_DIR}/cnmem.cpp")

# override rpath for rmm
set_target_properties(rmm PROPERTIES BUILD_RPATH "\$ORIGIN")

target_include_directories(rmm PUBLIC
  "${THRUST_INCLUDE_DIR}"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  )

# In order to find cnmem.h while building the library, we need this.
# For library users it is replaced by a copy of the same cnmem.h under rmm/detail
target_include_directories(rmm PRIVATE
  "${CNMEM_INCLUDE_DIR}"
  )

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

if(CUDA_STATIC_RUNTIME)
  message(STATUS "Enabling static linking of cudart")
  target_link_libraries(rmm PUBLIC CUDA::cudart_static)
else()
  target_link_libraries(rmm PUBLIC CUDA::cudart)
endif(CUDA_STATIC_RUNTIME)

if(USE_NVTX)
  message(STATUS "Using Nvidia Tools Extension")
else()
  target_compile_definitions(rmm PUBLIC NVTX_DISABLE)
endif(USE_NVTX)

# Set a default logging level if none was specified
set(DEFAULT_LOGGING_LEVEL OFF)

if(NOT LOGGING_LEVEL)
    message(STATUS "Setting logging level to '${DEFAULT_LOGGING_LEVEL}' since none specified.")
    set(LOGGING_LEVEL "${DEFAULT_LOGGING_LEVEL}" CACHE
      STRING "Choose the logging level." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE LOGGING_LEVEL PROPERTY STRINGS
        "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
else()
    message(STATUS "Setting logging level to '${LOGGING_LEVEL}'")
endif(NOT LOGGING_LEVEL)

target_compile_definitions(rmm PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGGING_LEVEL})

target_link_libraries(rmm PUBLIC spdlog::spdlog_header_only)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS rmm
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmm
        DESTINATION include)

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT RMM_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(rmm_doc DEPENDS RMM_DOXYGEN)
