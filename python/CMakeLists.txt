# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

# TODO: Pull this directly from the file written by rapids_cmake_write_version_file.
set(rmm_version 22.04.00)

project(
  rmm-python
  VERSION ${rmm_version}
  # TODO: Building Python extension modules via the python_extension_module requires the C language
  # to be enabled here. The test project that is built in scikit-build to verify various linking
  # options for the python library is hardcoded to build with C, so until that is fixed we need to
  # keep this.
  LANGUAGES C CXX CUDA)

option(FIND_RMM_CPP "Search for existing RMM CPP installations before defaulting to local files"
       OFF)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

# TODO: Should we guard this based on a scikit-build version? Override this function to avoid
# scikit-build clobbering symbol visibility.
function(_set_python_extension_symbol_visibility _target)
  if(PYTHON_VERSION_MAJOR VERSION_GREATER 2)
    set(_modinit_prefix "PyInit_")
  else()
    set(_modinit_prefix "init")
  endif()
  message("_modinit_prefix:${_modinit_prefix}")
  if("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set_target_properties(${_target} PROPERTIES LINK_FLAGS "/EXPORT:${_modinit_prefix}${_target}")
  elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(_script_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}-version-script.map)
    file(
      WRITE ${_script_path}
      # Note: The change is to this script, which does not indiscriminately mark all non PyInit
      # symbols as local.
      "{global: ${_modinit_prefix}${_target}; };")
    set_property(
      TARGET ${_target}
      APPEND_STRING
      PROPERTY LINK_FLAGS " -Wl,--version-script=\"${_script_path}\"")
  endif()
endfunction()

# Ignore unused variable warning.
set(ignored_variable "${SKBUILD}")

# If the user requested it we attempt to find RMM. TODO: Should we allow the user to specify a path
# instead of just searching? This version assumes that the installation has been appropriately
# configured for CMake discovery.
if(FIND_RMM_CPP)
  find_package(RMM ${rmm_version})
else()
  set(RMM_FOUND OFF)
endif(FIND_RMM_CPP)

if(NOT RMM_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  add_subdirectory(../ rmm-cpp)
endif(NOT RMM_FOUND)

add_subdirectory(cmake)
add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/_lib)
