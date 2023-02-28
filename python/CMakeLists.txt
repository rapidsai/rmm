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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(rmm_version 24.02.00)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.02/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

project(
  rmm-python
  VERSION ${rmm_version}
  LANGUAGES CXX)

option(FIND_RMM_CPP "Search for existing RMM C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find RMM.
if(FIND_RMM_CPP)
  find_package(rmm ${rmm_version})
else()
  set(rmm_FOUND OFF)
endif()

if(NOT rmm_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(CUDA_STATIC_RUNTIME ON)

  add_subdirectory(../ rmm-cpp EXCLUDE_FROM_ALL)
endif()

include(rapids-cython-core)
rapids_cython_init()

add_compile_definitions("SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/_lib)
