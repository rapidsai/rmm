# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(rmm_abi_namespace_consumer LANGUAGES CXX)

foreach(required_path IN ITEMS PREVIOUS_RMM_SOURCE_DIR PREVIOUS_RMM_BUILD_DIR PREVIOUS_RMM_LIBRARY)
  if(NOT DEFINED ${required_path} OR NOT EXISTS "${${required_path}}")
    message(FATAL_ERROR "${required_path} must name an existing path")
  endif()
endforeach()

find_package(rmm REQUIRED CONFIG)

add_library(previous_rmm STATIC IMPORTED)
set_target_properties(previous_rmm PROPERTIES IMPORTED_LOCATION "${PREVIOUS_RMM_LIBRARY}")

get_target_property(rmm_link_libraries rmm::rmm INTERFACE_LINK_LIBRARIES)
if(NOT rmm_link_libraries)
  message(FATAL_ERROR "rmm::rmm has no interface link libraries")
endif()

add_library(previous_rmm_call OBJECT previous.cpp)
target_include_directories(previous_rmm_call BEFORE PRIVATE "${PREVIOUS_RMM_SOURCE_DIR}/include"
                                                            "${PREVIOUS_RMM_BUILD_DIR}/include")
target_link_libraries(previous_rmm_call PRIVATE ${rmm_link_libraries})
set_target_properties(
  previous_rmm_call
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON)

add_library(rmm_abi_namespace_consumer SHARED current.cpp $<TARGET_OBJECTS:previous_rmm_call>)
target_link_libraries(
  rmm_abi_namespace_consumer
  PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rmm::rmm>" "$<LINK_LIBRARY:WHOLE_ARCHIVE,previous_rmm>"
          ${rmm_link_libraries})
set_property(TARGET rmm_abi_namespace_consumer PROPERTY CXX_STANDARD 20)
set_property(TARGET rmm_abi_namespace_consumer PROPERTY CXX_STANDARD_REQUIRED ON)

add_library(current_state_a SHARED current_state_a.cpp)
add_library(current_state_b SHARED current_state_b.cpp)
foreach(target IN ITEMS current_state_a current_state_b)
  target_link_libraries(${target} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rmm::rmm>")
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
endforeach()

add_library(previous_state SHARED previous_state.cpp)
target_include_directories(previous_state BEFORE PRIVATE "${PREVIOUS_RMM_SOURCE_DIR}/include"
                                                         "${PREVIOUS_RMM_BUILD_DIR}/include")
target_link_libraries(previous_state PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,previous_rmm>"
                                             ${rmm_link_libraries})
set_target_properties(previous_state PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

add_executable(rmm_abi_state_test state_test.cpp)
target_link_libraries(rmm_abi_state_test PRIVATE current_state_a current_state_b previous_state)
set_target_properties(rmm_abi_state_test PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
