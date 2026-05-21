# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

project(
  rmm-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX)

find_package(rmm "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

# pass through logging level to spdlog
add_compile_definitions("RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${RMM_LOGGING_LEVEL}")

add_subdirectory(rmm/_cuda)
add_subdirectory(rmm/librmm)
add_subdirectory(rmm/pylibrmm)
