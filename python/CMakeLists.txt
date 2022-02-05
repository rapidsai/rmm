#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

# TODO: Pull this directly from the file written by rapids_cmake_write_version_file.
set(rmm_version 22.04.00)

# TODO: What name do we want to use here?
project(rmm-python VERSION ${rmm_version} LANGUAGES CXX CUDA)

find_package(PythonExtensions REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Cython REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TODO: Is this the expected behavior? To use existing installs if found, otherwise the headers from the source directory?
# Try to find an installed RMM, if not found add the parent directory
# TODO (kk): Do some basic validation to ensure the parent is as expected
find_package(RMM ${rmm_version})
if(NOT RMM_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  add_subdirectory(../ rmm-cpp)
endif(NOT RMM_FOUND)

add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/_lib)
