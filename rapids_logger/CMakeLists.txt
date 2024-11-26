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

# cmake-lint: disable=R0915

include_guard(GLOBAL)

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

include(../cmake/thirdparty/get_spdlog.cmake)

#[=======================================================================[.rst:
rapids_make_logger
------------------

Generate a logger implementation customized for the specified namespace.

.. code-block:: cmake

  rapids_make_logger(<logger_namespace>
    [EXPORT_SET <export-set-name>]
    [LOGGER_TARGET <logger-target-name>]
    [LOGGER_HEADER_DIR <header-dir>]
    [LOGGER_MACRO_PREFIX <macro-prefix>]
  )

This function produces an interface target named <logger-target-name> that, when linked to by other targets, provides them a header file that defines a logger interface for the specified namespace. The logger is generated from the provided template files and is configured to use the specified namespace and macro prefix. The generated logger is placed in the specified header directory.

The logger implementation lives in a separate header file that is not included in the declaration header. The logger implementation is compiled into a separate target that must be linked to by any target that uses the logger.


``logger_namespace``
  The namespace for which to generate the logger implementation.

``EXPORT_SET``
  The name of the export set to which the logger target should be added. If not specified, the logger target is not added to any export set.

``LOGGER_TARGET``
  The name of the logger (and logger impl) target to create. If not specified, <logger-target-name> defaults to <logger_namespace>_logger.

``LOGGER_HEADER_DIR``
  The directory in which to place the generated logger header file. If not specified, the logger header file is placed in include/<logger_namespace>.

``LOGGER_MACRO_PREFIX``
  The prefix to use for the logger macros. If not specified, the macro prefix is the uppercase version of the logger namespace.

Result Targets
^^^^^^^^^^^^^^^^
  <logger-target-name> is an interface target that provides the logger interface for the specified namespace.

  <logger-target-name>_impl is an interface target that provides the logger implementation for the specified namespace. This target must be linked to by any target that uses the logger. Targets linking to this target will have the logger implementation compiled into them.

Examples
^^^^^^^^

Example on how to use :cmake:command:`rapids_make_logger`.


.. code-block:: cmake

  # Generate a logger for the namespace "rapids" and associate it with the
  # export set "rapids-exports".
  rapids_make_logger(rapids
    EXPORT_SET rapids-exports
  )

  # Generate a logger for the namespace "rmm" that does not support logging.
  rapids_make_logger(rapids)


#]=======================================================================]
function(rapids_make_logger logger_namespace)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids_make_logger")

  set(_rapids_options)
  set(_rapids_one_value EXPORT_SET LOGGER_TARGET LOGGER_HEADER_DIR LOGGER_MACRO_PREFIX)
  set(_rapids_multi_value)
  cmake_parse_arguments(_RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}" ${ARGN})

  # Most arguments are optional and can be inferred from the namespace by default.
  set(_RAPIDS_LOGGER_NAMESPACE ${logger_namespace})
  if(NOT _RAPIDS_LOGGER_TARGET)
    set(_RAPIDS_LOGGER_TARGET "${logger_namespace}_logger")
  endif()
  if(NOT _RAPIDS_LOGGER_HEADER_DIR)
    set(_RAPIDS_LOGGER_HEADER_DIR "include/${logger_namespace}")
  endif()
  if(NOT _RAPIDS_LOGGER_MACRO_PREFIX)
    string(TOUPPER ${logger_namespace} _RAPIDS_LOGGER_MACRO_PREFIX)
  endif()

  # All paths are computed relative to the current source/binary dir of the file from which the
  # function is invoked. As a result we cannot use relative paths here because CMake will root these
  # paths incorrectly for configure_file/install.
  set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/${_RAPIDS_LOGGER_HEADER_DIR})
  # TODO: Verify that installation works correctly with prefix removed.
  set(INSTALL_DIR ${_RAPIDS_LOGGER_HEADER_DIR})

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
  target_compile_features(${_RAPIDS_LOGGER_TARGET} INTERFACE cxx_std_17)

  # Create an interface target that will trigger compilation of the logger implementation in any
  # target that is linked to it.
  set(LOGGER_IMPL_SRC_OUTPUT_FILE ${BUILD_DIR}/logger_impl/logger.cpp)
  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/logger.cpp.in ${LOGGER_IMPL_SRC_OUTPUT_FILE})
  install(FILES ${LOGGER_IMPL_SRC_OUTPUT_FILE} DESTINATION ${INSTALL_DIR}/logger_impl)

  # Note that we cannot specify the source files directly in add_library, see the CMake
  # documentation explaining that these do not populate INTERFACE_SOURCES.
  # https://cmake.org/cmake/help/latest/command/add_library.html#interface-with-sources
  set(impl_target ${_RAPIDS_LOGGER_TARGET}_impl)
  add_library(${impl_target} INTERFACE)
  target_sources(
    ${impl_target}
    INTERFACE $<BUILD_INTERFACE:${LOGGER_IMPL_SRC_OUTPUT_FILE}>
              $<INSTALL_INTERFACE:${_RAPIDS_LOGGER_HEADER_DIR}/logger_impl/logger.cpp>)
  target_link_libraries(${impl_target} INTERFACE ${_RAPIDS_LOGGER_TARGET}
                                                 spdlog::spdlog_header_only)
  set_target_properties(${impl_target} PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                  INTERFACE_POSITION_INDEPENDENT_CODE ON)

  set(_install_export)
  if(_RAPIDS_EXPORT_SET)
    set(_install_export EXPORT ${_RAPIDS_EXPORT_SET})
  endif()

  install(TARGETS ${_RAPIDS_LOGGER_TARGET} ${_install_export})
  if(TARGET ${impl_target})
    install(TARGETS ${impl_target} ${_install_export})
  endif()

endfunction()
