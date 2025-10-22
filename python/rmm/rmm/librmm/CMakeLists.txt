# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources _logger.pyx)
set(linked_libraries rmm::rmm)

# Build all of the Cython targets
rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}"
                                                                               CXX)

# mark all symbols in these Cython targets "hidden" by default, so they won't collide with symbols
# loaded from other DSOs
foreach(_cython_target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${_cython_target} PROPERTIES C_VISIBILITY_PRESET hidden
                                                     CXX_VISIBILITY_PRESET hidden)
endforeach()

add_library(_torch_allocator SHARED _torch_allocator.cpp)
# Want the output to be called _torch_allocator.so
set_target_properties(_torch_allocator PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(_torch_allocator PRIVATE rmm::rmm)
cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
           OUTPUT_VARIABLE _torch_allocator_location)
install(TARGETS _torch_allocator DESTINATION "${_torch_allocator_location}")
