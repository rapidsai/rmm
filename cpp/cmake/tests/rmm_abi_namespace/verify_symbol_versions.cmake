# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

foreach(required_file IN ITEMS READELF RMM_LIBRARY PREVIOUS_RMM_LIBRARY CONSUMER_LIBRARY)
  if(NOT DEFINED ${required_file} OR NOT EXISTS "${${required_file}}")
    message(FATAL_ERROR "${required_file} must name an existing file")
  endif()
endforeach()
foreach(required_value IN ITEMS CURRENT_RMM_ABI_NAMESPACE PREVIOUS_RMM_ABI_NAMESPACE)
  if(NOT DEFINED ${required_value} OR "${${required_value}}" STREQUAL "")
    message(FATAL_ERROR "${required_value} must not be empty")
  endif()
endforeach()
if(CURRENT_RMM_ABI_NAMESPACE STREQUAL PREVIOUS_RMM_ABI_NAMESPACE)
  message(FATAL_ERROR "CURRENT_RMM_ABI_NAMESPACE and PREVIOUS_RMM_ABI_NAMESPACE must differ")
endif()

# Read demangled ELF symbols from library into output_variable.
function(read_symbols library output_variable)
  execute_process(
    COMMAND "${READELF}" --symbols --wide --demangle "${library}"
    OUTPUT_VARIABLE symbols
    ERROR_VARIABLE error_output
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "readelf failed for ${library}: ${error_output}")
  endif()
  set(${output_variable}
      "${symbols}"
      PARENT_SCOPE)
endfunction()

# Read demangled dynamic ELF symbols from library into output_variable.
function(read_dynamic_symbols library output_variable)
  execute_process(
    COMMAND "${READELF}" --dyn-syms --wide --demangle "${library}"
    OUTPUT_VARIABLE symbols
    ERROR_VARIABLE error_output
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "readelf failed for ${library}: ${error_output}")
  endif()
  set(${output_variable}
      "${symbols}"
      PARENT_SCOPE)
endfunction()

read_symbols("${RMM_LIBRARY}" rmm_symbols)
read_symbols("${PREVIOUS_RMM_LIBRARY}" previous_rmm_symbols)
read_symbols("${CONSUMER_LIBRARY}" consumer_symbols)
read_dynamic_symbols("${CONSUMER_LIBRARY}" consumer_dynamic_symbols)

set(current_align_up
    "FUNC[ \t]+GLOBAL[ \t]+DEFAULT[ \t]+[0-9]+[ \t]+rmm::${CURRENT_RMM_ABI_NAMESPACE}::align_up\\(")
set(previous_align_up
    "FUNC[ \t]+GLOBAL[ \t]+DEFAULT[ \t]+[0-9]+[ \t]+rmm::${PREVIOUS_RMM_ABI_NAMESPACE}::align_up\\("
)
set(unversioned_align_up "FUNC[ \t]+GLOBAL[ \t]+DEFAULT[ \t]+[0-9]+[ \t]+rmm::align_up\\(")
set(current_resource_map
    "OBJECT[ \t]+UNIQUE[ \t]+DEFAULT[ \t]+[0-9]+[ \t]+rmm::${CURRENT_RMM_ABI_NAMESPACE}::mr::detail::get_ref_map\\(\\)::device_id_to_resource"
)
set(previous_resource_map
    "OBJECT[ \t]+UNIQUE[ \t]+DEFAULT[ \t]+[0-9]+[ \t]+rmm::${PREVIOUS_RMM_ABI_NAMESPACE}::mr::detail::get_ref_map\\(\\)::device_id_to_resource"
)

if(NOT rmm_symbols MATCHES "${current_align_up}" OR NOT rmm_symbols MATCHES
                                                    "${current_resource_map}")
  message(FATAL_ERROR "Current static RMM does not contain the expected ABI-versioned symbols")
endif()
if(NOT previous_rmm_symbols MATCHES "${previous_align_up}" OR NOT previous_rmm_symbols MATCHES
                                                              "${previous_resource_map}")
  message(FATAL_ERROR "Previous static RMM does not contain the expected ABI-versioned symbols")
endif()
if(rmm_symbols MATCHES "${unversioned_align_up}" OR previous_rmm_symbols MATCHES
                                                    "${unversioned_align_up}")
  message(FATAL_ERROR "A static RMM library contains an unversioned rmm::align_up symbol")
endif()
foreach(expected_symbol IN ITEMS current_align_up previous_align_up current_resource_map
                                 previous_resource_map)
  if(NOT consumer_symbols MATCHES "${${expected_symbol}}")
    message(FATAL_ERROR "Consumer DSO is missing ${expected_symbol}")
  endif()
endforeach()
foreach(expected_symbol IN ITEMS current_align_up previous_align_up)
  if(NOT consumer_dynamic_symbols MATCHES "${${expected_symbol}}")
    message(FATAL_ERROR "Consumer DSO does not export ${expected_symbol}")
  endif()
endforeach()
