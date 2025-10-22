# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone NVTX3
function(find_and_configure_nvtx3)

  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)
  rapids_cpm_nvtx3(BUILD_EXPORT_SET rmm-exports INSTALL_EXPORT_SET rmm-exports)

endfunction()

find_and_configure_nvtx3()
