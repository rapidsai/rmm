# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

include(../../rapids_config.cmake)

project(
  rmm-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX)

find_package(rmm "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

# pass through logging level to spdlog
add_compile_definitions("SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Create an object library for the logger so that we don't have to recompile it. This assumes that
# we always have logging enabled when we built the C++ library since the targets that are created
# when we find_package(rmm) already have that setting baked in. TODO: Figure out how to use
# define_property in the C++ build to make SUPPORTS_LOGGING a queryable property that can be checked
# here. Otherwise we just assume that the Python package is never built against a C++ rmm without
# logging support.
add_library(cpp_logger OBJECT)
target_link_libraries(cpp_logger PRIVATE rmm::rmm_logger_impl)

add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/librmm)
add_subdirectory(rmm/pylibrmm)
