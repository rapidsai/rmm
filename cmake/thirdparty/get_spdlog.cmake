# =============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# Use CPM to find or clone speedlog
function(find_and_configure_spdlog VERSION)

  if(TARGET spdlog::spdlog_header_only)
    return()
  endif()

  rapids_cpm_find(
    spdlog ${VERSION}
    CPM_ARGS
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
    OPTIONS "SPDLOG_INSTALL TRUE")
  # spdlog
  if(spdlog_ADDED)
    install(TARGETS spdlog_header_only EXPORT rmm-exports)
  else()
    rapids_export_package(BUILD spdlog rmm-exports)
    rapids_export_package(INSTALL spdlog rmm-exports)
  endif()
endfunction()

find_and_configure_spdlog(1.8.5)
