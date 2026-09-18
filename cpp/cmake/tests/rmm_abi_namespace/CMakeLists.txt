# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(rmm_abi_namespace_test LANGUAGES CXX)

set(RMM_REPOSITORY_DIR
    ""
    CACHE PATH "Path to the RMM repository root")
if(NOT RMM_REPOSITORY_DIR)
  cmake_path(SET RMM_REPOSITORY_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../../..")
endif()

# Normalize a two-digit RMM version component for use in a C++ identifier.
function(normalize_version_component input output)
  string(REGEX REPLACE "^0+" "" normalized "${input}")
  if(normalized STREQUAL "")
    set(normalized 0)
  endif()
  set(${output}
      "${normalized}"
      PARENT_SCOPE)
endfunction()

file(STRINGS "${RMM_REPOSITORY_DIR}/VERSION" current_rmm_version LIMIT_COUNT 1)
if(NOT current_rmm_version MATCHES "^([0-9]+)\\.([0-9]+)\\.")
  message(FATAL_ERROR "Unable to parse RMM version: ${current_rmm_version}")
endif()
normalize_version_component("${CMAKE_MATCH_1}" current_rmm_version_major)
normalize_version_component("${CMAKE_MATCH_2}" current_rmm_version_minor)
set(current_rmm_abi_namespace "_RMM_${current_rmm_version_major}_${current_rmm_version_minor}")

# rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
set(previous_rmm_version "26.06.00")
set(previous_rmm_abi_namespace "_RMM_26_6")
if(current_rmm_abi_namespace STREQUAL previous_rmm_abi_namespace)
  message(FATAL_ERROR "The current and previous test ABI namespaces must be different")
endif()

set(previous_rmm_source_dir "${CMAKE_CURRENT_BINARY_DIR}/previous-rmm-source")
file(REMOVE_RECURSE "${previous_rmm_source_dir}")
file(MAKE_DIRECTORY "${previous_rmm_source_dir}")
file(
  COPY "${RMM_REPOSITORY_DIR}/cpp"
  DESTINATION "${previous_rmm_source_dir}"
  PATTERN "build" EXCLUDE)
file(COPY "${RMM_REPOSITORY_DIR}/cmake" "${RMM_REPOSITORY_DIR}/RAPIDS_BRANCH"
     DESTINATION "${previous_rmm_source_dir}")
file(WRITE "${previous_rmm_source_dir}/VERSION" "${previous_rmm_version}\n")

set(rmm_build_dir "${CMAKE_CURRENT_BINARY_DIR}/rmm-build")
set(previous_rmm_build_dir "${CMAKE_CURRENT_BINARY_DIR}/previous-rmm-build")
set(consumer_build_dir "${CMAKE_CURRENT_BINARY_DIR}/consumer-build")
set(consumer_source_dir "${CMAKE_CURRENT_LIST_DIR}/consumer")
set(rmm_library "${rmm_build_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}rmm${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(previous_rmm_library
    "${previous_rmm_build_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}rmm${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(consumer_library
    "${consumer_build_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}rmm_abi_namespace_consumer${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
set(state_test "${consumer_build_dir}/rmm_abi_state_test${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_target(
  verify_rmm_abi_namespace ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${rmm_build_dir}" "${previous_rmm_build_dir}"
          "${consumer_build_dir}"
  COMMAND
    ${CMAKE_COMMAND} -S "${RMM_REPOSITORY_DIR}/cpp" -B "${rmm_build_dir}" -G "${CMAKE_GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
  COMMAND ${CMAKE_COMMAND} --build "${rmm_build_dir}" --target rmm
  COMMAND
    ${CMAKE_COMMAND} -S "${previous_rmm_source_dir}/cpp" -B "${previous_rmm_build_dir}" -G
    "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF
    -DBUILD_BENCHMARKS=OFF
  COMMAND ${CMAKE_COMMAND} --build "${previous_rmm_build_dir}" --target rmm
  COMMAND
    ${CMAKE_COMMAND} -S "${consumer_source_dir}" -B "${consumer_build_dir}" -G "${CMAKE_GENERATOR}"
    "-Drmm_DIR=${rmm_build_dir}" "-DPREVIOUS_RMM_SOURCE_DIR=${previous_rmm_source_dir}/cpp"
    "-DPREVIOUS_RMM_BUILD_DIR=${previous_rmm_build_dir}"
    "-DPREVIOUS_RMM_LIBRARY=${previous_rmm_library}"
  COMMAND ${CMAKE_COMMAND} --build "${consumer_build_dir}"
  COMMAND
    ${CMAKE_COMMAND} "-DREADELF=${CMAKE_READELF}" "-DRMM_LIBRARY=${rmm_library}"
    "-DPREVIOUS_RMM_LIBRARY=${previous_rmm_library}" "-DCONSUMER_LIBRARY=${consumer_library}"
    "-DCURRENT_RMM_ABI_NAMESPACE=${current_rmm_abi_namespace}"
    "-DPREVIOUS_RMM_ABI_NAMESPACE=${previous_rmm_abi_namespace}" -P
    "${CMAKE_CURRENT_LIST_DIR}/verify_symbol_versions.cmake"
  COMMAND "${state_test}"
  COMMENT "Verifying composition and process state for multiple RMM ABI versions"
  VERBATIM)
