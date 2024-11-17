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

# Use CPM to find or clone speedlog TODO: The logic here should be upstreamed into rapids-logger and
# probably removed from rapids-cmake. Note that it is not possible to build with two different modes
# for different projects in the same build environment, so this functionality must be kept
# independent of the rapids_make_logger function. Alternatively, we could update the cpm calls for
# spdlog to not promote targets to global scope and instead allow each project to find and configure
# spdlog independently.
function(find_and_configure_spdlog)

  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  if(RMM_SPDLOG_DYNAMIC)
    rapids_cpm_spdlog(
      INSTALL_EXPORT_SET rmm-exports BUILD_EXPORT_SET rmm-exports OPTIONS "SPDLOG_BUILD_SHARED ON"
                                                                  "BUILD_SHARED_LIBS ON")
  else()
    # For static spdlog usage assume that we want to hide as many symbols as possible, so do not use
    # pre-built libraries. It's quick enough to build that there is no real benefit to supporting
    # the alternative.
    set(CPM_DOWNLOAD_spdlog ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
    rapids_cpm_spdlog(
      INSTALL_EXPORT_SET rmm-exports
      BUILD_EXPORT_SET rmm-exports OPTIONS "SPDLOG_BUILD_SHARED OFF" "BUILD_SHARED_LIBS OFF"
                                   EXCLUDE_FROM_ALL)
    set_target_properties(
      spdlog
      PROPERTIES CXX_VISIBILITY_PRESET hidden
                 VISIBILITY_INLINES_HIDDEN ON
                 POSITION_INDEPENDENT_CODE ON)
  endif()
endfunction()

find_and_configure_spdlog()
