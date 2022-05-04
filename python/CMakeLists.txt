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

set(rmm_version 22.06.00)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.06/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

project(
  rmm-python
  VERSION ${rmm_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C
            CXX
            # TODO: The C++ RMM CMake configuration targets cuda_std_17 features, which prior to
            # CMake 3.22 will also pull in the corresponding required languages even if this project
            # does not actually require those languages. As a result, we need to include CUDA here.
            # We can remove CUDA once we upgrade the minimum required CMake version to 3.22.
            CUDA)

option(FIND_RMM_CPP "Search for existing RMM C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find RMM.
if(FIND_RMM_CPP)
  find_package(RMM ${rmm_version})
else()
  set(RMM_FOUND OFF)
endif()

if(NOT RMM_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  add_subdirectory(../ rmm-cpp)
endif()

include(rapids-cython)
rapids_cython_init()
add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/_lib)
