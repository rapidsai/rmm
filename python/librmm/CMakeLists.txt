# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

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

add_subdirectory(../../cpp rmm-cpp)
