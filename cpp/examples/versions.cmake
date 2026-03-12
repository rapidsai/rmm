# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use STRINGS to trim whitespace/newlines
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../RAPIDS_BRANCH" RMM_TAG)
if(NOT RMM_TAG)
  message(
    FATAL_ERROR
      "Could not determine branch name to use for checking out rapids-cmake. The file \"${CMAKE_CURRENT_LIST_DIR}/../../RAPIDS_BRANCH\" is missing."
  )
endif()
