# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

# Use CPM to find or clone spdlog.
function(find_and_configure_spdlog)

  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  rapids_cpm_spdlog(
    FMT_OPTION "EXTERNAL_FMT_HO"
    BUILD_EXPORT_SET rmm-exports
    INSTALL_EXPORT_SET rmm-exports)

  include(${rapids-cmake-dir}/export/cpm.cmake)
  include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)
  rapids_cpm_package_details(spdlog version repository tag shallow exclude)

  # TODO: Switch to EXCLUDE_FROM_ALL once we have a way to handle the fmt linkage issue. TODO: We
  # shouldn't have to repeat all the information from inside rapids-cmake here. Fixing that may have
  # to wait until we upstream rapids-logger logic to rapids-cmake, though.
  rapids_export_cpm(
    BUILD spdlog rmm-exports DEFAULT_DOWNLOAD_OPTION "RMM_DOWNLOAD_SPDLOG"
    CPM_ARGS NAME
             spdlog
             VERSION
             ${version}
             GIT_REPOSITORY
             ${repository}
             GIT_TAG
             ${tag}
             GIT_SHALLOW
             ${shallow}
             EXCLUDE_FROM_ALL
             ${exclude}
             OPTIONS
             "SPDLOG_INSTALL ON"
             "SPDLOG_FMT_EXTERNAL_HO ON")
  rapids_export_cpm(
    BUILD spdlog rmm-exports DEFAULT_DOWNLOAD_OPTION "RMM_DOWNLOAD_SPDLOG"
    CPM_ARGS NAME
             spdlog
             VERSION
             ${version}
             GIT_REPOSITORY
             ${repository}
             GIT_TAG
             ${tag}
             GIT_SHALLOW
             ${shallow}
             EXCLUDE_FROM_ALL
             ${exclude}
             OPTIONS
             "SPDLOG_INSTALL ON"
             "SPDLOG_FMT_EXTERNAL_HO ON")
endfunction()

find_and_configure_spdlog()
