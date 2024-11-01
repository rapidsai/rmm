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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../rapids_config.cmake)

include(rapids-cmake)
include(rapids-cpm)

project(
  RAPIDS_LOGGER
  VERSION 0.0.1
  LANGUAGES CXX)

rapids_cmake_build_type(Release)

rapids_cpm_init()

# Make sure install logic is handled correctly, namely that nothing is installed from these. I think
# we'll need to both not specify an export set and EXCLUDE_FROM_ALL.
include(../cmake/thirdparty/get_fmt.cmake)
include(../cmake/thirdparty/get_spdlog.cmake)

# Generate and install the logger files
function(rapids_make_logger)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids_make_logger")

  set(_rapids_options RMM_BACKWARDS_COMPATIBILITY SUPPORTS_LOGGING)
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

  set(LOGGER_OUTPUT_FILE ${BUILD_DIR}/logger.hpp)
  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger.hpp.in ${LOGGER_OUTPUT_FILE})
  install(FILES ${LOGGER_OUTPUT_FILE} DESTINATION ${INSTALL_DIR})

  set(LOGGER_IMPL_OUTPUT_FILE ${BUILD_DIR}/logger_impl/logger_impl.hpp)
  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger_impl.hpp.in ${LOGGER_IMPL_OUTPUT_FILE})
  install(FILES ${LOGGER_IMPL_OUTPUT_FILE} DESTINATION ${INSTALL_DIR}/logger_impl)

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
  if(_RAPIDS_RMM_BACKWARDS_COMPATIBILITY)
    if(NOT _RAPIDS_SUPPORTS_LOGGING)
      message(STATUS "RMM_BACKWARDS_COMPATIBILITY requires SUPPORTS_LOGGING, turning it on")
      set(_RAPIDS_SUPPORTS_LOGGING ON)
    endif()
    target_compile_definitions(${_RAPIDS_LOGGER_TARGET} INTERFACE RMM_BACKWARDS_COMPATIBILITY)
  endif()
  if(_RAPIDS_SUPPORTS_LOGGING)
    target_compile_definitions(${_RAPIDS_LOGGER_TARGET} INTERFACE SUPPORTS_LOGGING)

    if(NOT _RAPIDS_RMM_BACKWARDS_COMPATIBILITY)
      # Create an interface target that will trigger compilation of the logger implementation in any
      # target that is linked to it.
      set(LOGGER_IMPL_SRC_OUTPUT_FILE ${BUILD_DIR}/logger_impl/logger.cpp)
      configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger.cpp.in
                     ${LOGGER_IMPL_SRC_OUTPUT_FILE})
      install(FILES ${LOGGER_IMPL_SRC_OUTPUT_FILE} DESTINATION ${INSTALL_DIR}/logger_impl)

      # Note that we cannot specify the source files directly in add_library, see the CMake
      # documentation explaining that these do not populate INTERFACE_SOURCES.
      # https://cmake.org/cmake/help/latest/command/add_library.html#interface-with-sources
      add_library(${_RAPIDS_LOGGER_TARGET}_impl INTERFACE)
      target_sources(
        ${_RAPIDS_LOGGER_TARGET}_impl
        INTERFACE $<BUILD_INTERFACE:${LOGGER_IMPL_SRC_OUTPUT_FILE}>
                  $<INSTALL_INTERFACE:${_RAPIDS_LOGGER_HEADER_DIR}/logger_impl/logger.cpp>)
    endif()
  endif()

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
