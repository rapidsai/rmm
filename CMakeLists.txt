# =============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.18...3.18 FATAL_ERROR)

project(
  RMM
  VERSION 0.18.0
  LANGUAGES CXX)

include(cmake/Modules/CPM.cmake)
include(cmake/Modules/RMM_thirdparty.cmake)
include(cmake/Modules/Version.cmake)

# Write the version header
write_version()

# build type

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "RMM: Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# build options

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

# cudart can be statically linked or dynamically linked the python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

# find packages we depend on

find_package(CUDAToolkit REQUIRED)

# library targets

add_library(rmm INTERFACE)
add_library(rmm::rmm ALIAS rmm)

target_include_directories(
  rmm
  INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>" "$<INSTALL_INTERFACE:include>")

if(CUDA_STATIC_RUNTIME)
  message(STATUS "RMM: Enabling static linking of cudart")
  target_link_libraries(rmm INTERFACE CUDA::cudart_static)
else()
  target_link_libraries(rmm INTERFACE CUDA::cudart)
endif(CUDA_STATIC_RUNTIME)

target_link_libraries(rmm INTERFACE rmm::Thrust)
target_link_libraries(rmm INTERFACE spdlog::spdlog_header_only)
target_compile_features(rmm INTERFACE cxx_std_14 cuda_std_14)

# Set logging level. Must go before including gtests and benchmarks.

set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR"
                                              "CRITICAL" "OFF")
message(STATUS "RMM: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'")

# optionally build tests

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif(BUILD_TESTS)

# add google benchmark

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

# install targets

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/rmm)

install(TARGETS rmm EXPORT rmm-targets)

install(DIRECTORY include/rmm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${RMM_BINARY_DIR}/include/rmm/version_config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmm)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/rmm-config.cmake.in ${RMM_BINARY_DIR}/rmm-config.cmake
                              INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

write_basic_package_version_file(${RMM_BINARY_DIR}/rmm-config-version.cmake
                                 COMPATIBILITY SameMinorVersion)

install(
  EXPORT rmm-targets
  FILE rmm-targets.cmake
  NAMESPACE rmm::
  DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${RMM_BINARY_DIR}/rmm-config.cmake ${RMM_BINARY_DIR}/rmm-config-version.cmake
              ${RMM_SOURCE_DIR}/cmake/install/FindThrust.cmake DESTINATION ${INSTALL_CONFIGDIR})

# make documentation

add_custom_command(
  OUTPUT RMM_DOXYGEN
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for RMM doxygen docs")

add_custom_target(
  rmm_doc
  DEPENDS RMM_DOXYGEN
  COMMENT "Target for the custom command to build the RMM doxygen docs")
