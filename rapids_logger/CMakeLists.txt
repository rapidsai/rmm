# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

# TODO: Should this have a full CMakeLists capable of being built by itself, or should it assume
# that it's always being called from another CMakeLists? cmake_minimum_required(VERSION 3.26.4
# FATAL_ERROR)
#
# include(../rapids_config.cmake)
#
# include(rapids-cmake) include(rapids-cpm)
#
# project( RAPIDS_LOGGER VERSION 0.0.1 LANGUAGES CXX)
#
# rapids_cmake_write_version_file(include/rmm/version_config.hpp) rapids_cmake_build_type(Release)
#
# option(SUPPORTS_LOGGING "Whether to build with real logging support or not" OFF)
# option(RMM_BACKWARDS_COMPATIBILITY "Enable backwards compatibility with the legacy RMM logger"
# OFF) TODO: Change the default to rapids set(LOGGER_NAMESPACE rmm CACHE STRING "The namespace to
# use for the logger")
#
# message(VERBOSE "RMM: Build with NVTX support: ${RMM_NVTX}")
#
# rapids_cpm_init()
#
# Make sure install logic is handled correctly, namely that nothing is installed from these. I think
# we'll need to both not specify an export set and EXCLUDE_FROM_ALL.
# include(../cmake/thirdparty/get_fmt.cmake) include(../cmake/thirdparty/get_spdlog.cmake)

# Generate and install the logger files
function(rapids_make_logger)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids_make_logger")

  set(_rapids_options)
  # TODO: Check for required options
  set(_rapids_one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET LOGGER_TARGET LOGGER_HEADER_DIR
                        LOGGER_NAMESPACE LOGGER_MACRO_PREFIX CMAKE_NAMESPACE)
  set(_rapids_multi_value)
  cmake_parse_arguments(_RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}" ${ARGN})

  # All paths are computed relative to the current source/binary dir of the file from which the
  # function is invoked. As a result we cannot use relative paths here because CMake will root these
  # paths incorrectly for configure_file/install.
  set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_RAPIDS_LOGGER_HEADER_DIR})
  set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${_RAPIDS_LOGGER_HEADER_DIR})

  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger.hpp.in ${BUILD_DIR}/logger.hpp)
  install(FILES ${LOGGER_OUTPUT_FILE} DESTINATION ${INSTALL_DIR})

  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger_impl.hpp.in ${BUILD_DIR}/logger_impl.hpp)
  install(FILES ${LOGGER_IMPL_OUTPUT_FILE} DESTINATION ${INSTALL_DIR})

  add_library(${_RAPIDS_LOGGER_TARGET} INTERFACE)
  include(GNUInstallDirs)
  # Note: The BUILD_INTERFACE setting assumes that LOGGER_HEADER_DIR is the subdirectory of
  # CMAKE_INSTALL_INCLUDEDIR relative to which all includes are rooted in the C++ code files. I
  # think that is a safe assumption though since if it were violated then the INSTALL_INTERFACE
  # would not only be incorrect (if computed using LOGGER_HEADER_DIR), but it would also break
  # consumers of the installed package who expect to be able to write `#include
  # <${LOGGER_HEADER_DIR/include\//}/logger.hpp>` and have it work.
  target_include_directories(
    ${_RAPIDS_LOGGER_TARGET}
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>"
              "$<INSTALL_INTERFACE:${_RAPIDS_LOGGER_HEADER_DIR}>")
  if(_RAPIDS_LOGGER_CMAKE_NAMESPACE)
    set(CMAKE_NAMESPACE "NAMESPACE ${_RAPIDS_LOGGER_CMAKE_NAMESPACE}")
  endif()
  if(_RAPIDS_INSTALL_EXPORT_SET)
    install(
      TARGETS ${_RAPIDS_LOGGER_TARGET}
      EXPORT ${_RAPIDS_INSTALL_EXPORT_SET}
      ${CMAKE_NAMESPACE})
  endif()
  if(_RAPIDS_BUILD_EXPORT_SET)
    export(EXPORT ${_RAPIDS_BUILD_EXPORT_SET} ${CMAKE_NAMESPACE})
  endif()
endfunction()
