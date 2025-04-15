# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

include(../../cmake/rapids_config.cmake)

project(
  librmm-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX)

# Check if rmm is already available. If so, it's the user's responsibility to ensure that the CMake
# package is also available at build time of the Python rmm package.
find_package(rmm "${RAPIDS_VERSION}")

if(rmm_FOUND)
  return()
endif()

unset(rmm_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp rmm-cpp)
