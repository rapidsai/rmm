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

set(rmm_version 23.10.00)

include(../fetch_rapids.cmake)

project(
  rmm-python
  VERSION ${rmm_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX)

option(FIND_RMM_CPP "Search for existing RMM C++ installations before defaulting to local files"
       OFF)
option(RMM_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)

# If the user requested it we attempt to find RMM.
if(FIND_RMM_CPP)
  find_package(rmm ${rmm_version})
else()
  set(rmm_FOUND OFF)
endif()

if(NOT rmm_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  set(_exclude_from_all "")
  if(RMM_BUILD_WHEELS)
    # Statically link dependencies if building wheels
    set(CUDA_STATIC_RUNTIME ON)
    # Don't install the rmm C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../ rmm-cpp ${_exclude_from_all})
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/_lib)
