# =============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(rapids_config.cmake)

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

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR"
                                              "CRITICAL" "OFF")

message(VERBOSE "RMM: Build with NVTX support: ${USE_NVTX}")
# Set logging level. Must go before including gtests and benchmarks. Set the possible values of
# build type for cmake-gui.
message(STATUS "RMM: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'")

# cudart can be linked statically or dynamically
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

# ##################################################################################################
# * compiler options -------------------------------------------------------------------------------

# find packages we depend on
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET rmm-exports
  INSTALL_EXPORT_SET rmm-exports)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

include(cmake/thirdparty/get_fmt.cmake)
include(cmake/thirdparty/get_spdlog.cmake)
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_nvtx.cmake)

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

add_library(rmm INTERFACE)
add_library(rmm::rmm ALIAS rmm)

target_include_directories(rmm INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")

if(CUDA_STATIC_RUNTIME)
  message(STATUS "RMM: Enabling static linking of cudart")
  target_link_libraries(rmm INTERFACE CUDA::cudart_static)
  target_compile_definitions(rmm INTERFACE RMM_STATIC_CUDART)
else()
  target_link_libraries(rmm INTERFACE CUDA::cudart)
endif()

target_link_libraries(rmm INTERFACE CCCL::CCCL)
target_link_libraries(rmm INTERFACE fmt::fmt-header-only)
target_link_libraries(rmm INTERFACE spdlog::spdlog_header_only)
target_link_libraries(rmm INTERFACE dl)
target_link_libraries(rmm INTERFACE nvtx3::nvtx3-cpp)
target_compile_features(rmm INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
target_compile_definitions(rmm INTERFACE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(rmm INTERFACE NVTX_DISABLE)
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
  rapids_cpm_gbench()
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets --------------------------------------------------------------------------------

include(CPack)

# install export targets
install(TARGETS rmm EXPORT rmm-exports)
install(DIRECTORY include/rmm/ DESTINATION include/rmm)
install(FILES ${RMM_BINARY_DIR}/include/rmm/version_config.hpp DESTINATION include/rmm)

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
