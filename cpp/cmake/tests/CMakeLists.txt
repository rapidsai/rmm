# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(rmm_cmake_tests LANGUAGES NONE)

enable_testing()

set(RMM_REPOSITORY_DIR
    ""
    CACHE PATH "Path to the RMM repository root")
if(NOT RMM_REPOSITORY_DIR)
  cmake_path(SET RMM_REPOSITORY_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../..")
endif()

set(rmm_abi_namespace_build_dir "${CMAKE_CURRENT_BINARY_DIR}/rmm_abi_namespace")

add_test(
  NAME rmm_abi_namespace_configure
  COMMAND
    ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_LIST_DIR}/rmm_abi_namespace" -B
    "${rmm_abi_namespace_build_dir}" -G "${CMAKE_GENERATOR}"
    "-DRMM_REPOSITORY_DIR=${RMM_REPOSITORY_DIR}")

add_test(NAME rmm_abi_namespace COMMAND ${CMAKE_COMMAND} --build "${rmm_abi_namespace_build_dir}")

set_tests_properties(rmm_abi_namespace_configure PROPERTIES FIXTURES_SETUP rmm_abi_namespace)
set_tests_properties(rmm_abi_namespace PROPERTIES FIXTURES_REQUIRED rmm_abi_namespace)
