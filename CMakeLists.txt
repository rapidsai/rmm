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

project(RMM VERSION 0.17.0 LANGUAGES C CXX CUDA)

include(cmake/CPM.cmake)

###################################################################################################
# - build type

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
# - compiler options

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")
endif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

###################################################################################################
# - build options

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
if(DISABLE_DEPRECATION_WARNING)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNING)

# cudart can be statically linked or dynamically linked the python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

###################################################################################################
# find packages we depend on

find_package(CUDAToolkit REQUIRED)

###################################################################################################
# cmake modules

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

###################################################################################################
# third-party dependencies

include(RMM_thirdparty)

###################################################################################################
# library targets

add_library(rmm INTERFACE)
add_library(rmm::rmm ALIAS rmm)

target_include_directories(rmm INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  )

if(CUDA_STATIC_RUNTIME)
  message(STATUS "Enabling static linking of cudart")
  target_link_libraries(rmm INTERFACE CUDA::cudart_static)
else()
  target_link_libraries(rmm INTERFACE CUDA::cudart)
endif(CUDA_STATIC_RUNTIME)

target_link_libraries(rmm INTERFACE rmm::Thrust spdlog::spdlog_header_only ${CMAKE_DL_LIBS})

###################################################################################################
# Set logging level. Must go before including gtests and benchmarks. 

set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS
    "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

###################################################################################################
# add gtest

if(BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif(BUILD_TESTS)

###################################################################################################
# add google benchmark

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

###################################################################################################
# install targets

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/rmm)

install(TARGETS rmm
  EXPORT rmm-targets)

install(DIRECTORY include/rmm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/rmm-config.cmake.in
  ${RMM_BINARY_DIR}/rmm-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
  )

write_basic_package_version_file(
  ${RMM_BINARY_DIR}/rmm-config-version.cmake
  COMPATIBILITY SameMinorVersion
  )

install(EXPORT rmm-targets
  FILE rmm-targets.cmake
  NAMESPACE rmm::
  DESTINATION ${INSTALL_CONFIGDIR}
  )

install(FILES
  ${RMM_BINARY_DIR}/rmm-config.cmake
  ${RMM_BINARY_DIR}/rmm-config-version.cmake
  ${RMM_SOURCE_DIR}/cmake/install/FindThrust.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
  )
  
###################################################################################################
# make documentation

add_custom_command(OUTPUT RMM_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(rmm_doc DEPENDS RMM_DOXYGEN)
