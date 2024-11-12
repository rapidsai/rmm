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

# Use CPM to find or clone speedlog
function(find_and_configure_spdlog)

  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  # TODO: When we disable backwards compatibility mode we will switch these flags on to build a
  # static lib of spdlog and link to that. The current approach of using a header-only spdlog works
  # with both compatibility and new logger modes, but it does not support completely hiding spdlog
  # symbols due to the inlining of functions that spdlog does when in header-only mode. When we
  # switch, forcing a download is one way to rebuild and ensure that we can build with the necessary
  # symbol visibility flag. The other option is to pass Wl,--exclude-libs,libspdlog to the linker,
  # which seems to capture a couple of symbols that this setting misses. In either case we may need
  # to force building since spdlog's CMake reuses the same target for static and dynamic library
  # builds.

  set(_options)
  # cmake-format: off
  ## TODO: Figure out how to use this variable properly inside
  ## rapids_cpm_spdlog. The way quotes are being interpolated is almost
  ## certainly not what I expect.
  #set(_options OPTIONS "SPDLOG_BUILD_SHARED OFF" "BUILD_SHARED_LIBS OFF")
  #if(RMM_SPDLOG_DYNAMIC)
  #  set(_options OPTIONS "SPDLOG_BUILD_SHARED ON" "BUILD_SHARED_LIBS ON")
  #else()
  #  set(CPM_DOWNLOAD_spdlog ON)
  #  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
  #endif()
# cmake-format: on
  rapids_cpm_spdlog(
    # TODO: We can drop this when we stop using header-only spdlog.
    FMT_OPTION "EXTERNAL_FMT_HO"
    INSTALL_EXPORT_SET rmm-exports
    BUILD_EXPORT_SET rmm-exports # cmake-format: off
    #OPTIONS "SPDLOG_BUILD_SHARED OFF" "BUILD_SHARED_LIBS OFF"
    #EXCLUDE_FROM_ALL
# cmake-format: on
  )

endfunction()

find_and_configure_spdlog()
