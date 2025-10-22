# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources device_buffer.pyx logger.pyx cuda_stream.pyx helper.pyx stream.pyx)
set(linked_libraries rmm::rmm)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibrmm_ ASSOCIATED_TARGETS rmm)

# mark all symbols in these Cython targets "hidden" by default, so they won't collide with symbols
# loaded from other DSOs
foreach(_cython_target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${_cython_target} PROPERTIES C_VISIBILITY_PRESET hidden
                                                     CXX_VISIBILITY_PRESET hidden)
endforeach()

add_subdirectory(memory_resource)
