# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to find or clone fmt
function(find_and_configure_fmt)

  include(${rapids-cmake-dir}/cpm/fmt.cmake)
  rapids_cpm_fmt(INSTALL_EXPORT_SET rmm-exports)
  rapids_export_package(BUILD fmt rmm-exports)

  if(fmt_ADDED)
    rapids_export(
      BUILD fmt
      EXPORT_SET fmt-targets
      GLOBAL_TARGETS fmt fmt-header-only
      NAMESPACE fmt::)
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD fmt [=[${CMAKE_CURRENT_LIST_DIR}]=] rmm-exports)
  endif()
endfunction()

find_and_configure_fmt()
