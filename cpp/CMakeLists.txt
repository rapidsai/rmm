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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  RMM
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX)

# Write the version header
rapids_cmake_write_version_file(include/rmm/version_config.hpp)

# ##################################################################################################
# * build type -------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * build options ----------------------------------------------------------------------------------

option(RMM_NVTX "Build RMM with NVTX support" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
# This is mostly so that dependent libraries are configured in shared mode for downstream dependents
# of RMM that get their common dependencies transitively.
option(BUILD_SHARED_LIBS "Build RMM shared libraries" ON)
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR"
                                              "CRITICAL" "OFF")

message(VERBOSE "RMM: Build with NVTX support: ${RMM_NVTX}")
# Set logging level. Must go before including gtests and benchmarks. Set the possible values of
# build type for cmake-gui.
message(STATUS "RMM: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'")

# cudart can be linked statically or dynamically
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

# ##################################################################################################
# * compiler options -------------------------------------------------------------------------------

# For now, disable CMake's automatic module scanning for C++ files.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# find packages we depend on
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET rmm-exports
  INSTALL_EXPORT_SET rmm-exports)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET rmm-exports INSTALL_EXPORT_SET rmm-exports)
create_logger_macros(RMM "rmm::default_logger()" include/rmm)

include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_nvtx.cmake)

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

add_library(
  rmm
  src/aligned.cpp
  src/cuda_device.cpp
  src/cuda_stream_pool.cpp
  src/cuda_stream_view.cpp
  src/cuda_stream.cpp
  src/device_buffer.cpp
  src/error.cpp
  src/exec_policy.cpp
  src/logger.cpp
  src/prefetch.cpp)
add_library(rmm::rmm ALIAS rmm)

target_include_directories(
  rmm
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  INTERFACE "$<INSTALL_INTERFACE:include>")

if(CUDA_STATIC_RUNTIME)
  message(STATUS "RMM: Enabling static linking of cudart")
  target_link_libraries(rmm PUBLIC CUDA::cudart_static)
else()
  target_link_libraries(rmm PUBLIC CUDA::cudart)
endif()

target_link_libraries(rmm PUBLIC CCCL::CCCL ${CMAKE_DL_LIBS} nvtx3::nvtx3-cpp
                                 rapids_logger::rapids_logger)

set_target_properties(
  rmm
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_VISIBILITY_PRESET hidden
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(rmm PUBLIC LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# Enable NVTX if necessary
if(RMM_NVTX)
  target_compile_definitions(rmm PUBLIC RMM_NVTX)
endif()

# ##################################################################################################
# * tests and benchmarks ---------------------------------------------------------------------------

if((BUILD_TESTS OR BUILD_BENCHMARKS) AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(rapids-cuda)
  rapids_cuda_init_architectures(RMM)
  enable_language(CUDA)

  # Since RMM only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  include("${CMAKE_PROJECT_RMM_INCLUDE}")
  message(STATUS "RMM: Building benchmarks with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ##################################################################################################
# * add tests --------------------------------------------------------------------------------------

if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
  include(CTest) # calls enable_testing()

  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks ---------------------------------------------------------------------------------

if(BUILD_BENCHMARKS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets --------------------------------------------------------------------------------

include(GNUInstallDirs)
include(CPack)

# install export targets
install(TARGETS rmm EXPORT rmm-exports)
install(DIRECTORY include/rmm/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmm)
install(FILES ${RMM_BINARY_DIR}/include/rmm/version_config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmm)

set(doc_string
    [=[
Provide targets for RMM: RAPIDS Memory Manager.

The goal of the [RMM](https://github.com/rapidsai/rmm) is to provide:

  A common interface that allows customizing device and host memory allocation
  A collection of implementations of the interface
  A collection of data structures that use the interface for memory allocation
]=])

rapids_export(
  INSTALL rmm
  EXPORT_SET rmm-exports
  GLOBAL_TARGETS rmm
  NAMESPACE rmm::
  DOCUMENTATION doc_string)

# ##################################################################################################
# * build export -----------------------------------------------------------------------------------

rapids_export(
  BUILD rmm
  EXPORT_SET rmm-exports
  GLOBAL_TARGETS rmm
  NAMESPACE rmm::
  DOCUMENTATION doc_string)

# ##################################################################################################
# * make documentation -----------------------------------------------------------------------------

add_custom_command(
  OUTPUT RMM_DOXYGEN
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
  COMMAND ${CMAKE_COMMAND} -E env "RAPIDS_VERSION=${RAPIDS_VERSION}"
          "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}" doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for RMM doxygen docs")

add_custom_target(
  rmm_doc
  DEPENDS RMM_DOXYGEN
  COMMENT "Target for the custom command to build the RMM doxygen docs")

# ##################################################################################################
# * make gdb helper scripts ------------------------------------------------------------------------

# optionally assemble Thrust pretty-printers
if(CCCL_SOURCE_DIR)
  configure_file(scripts/load-pretty-printers.in load-pretty-printers @ONLY)
endif()
