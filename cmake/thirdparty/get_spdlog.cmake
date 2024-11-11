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
  # static lib of spdlog and link to that. Forcing a download is one way to rebuild and ensure that
  # we can build with the necessary symbol visibility flag. The other option is to pass
  # Wl,--exclude-libs,libspdlog to the linker, which seems to capture a couple of symbols that this
  # setting misses. However, in either case we may need to force building since spdlog's CMake
  # reuses the same target for static and dynamic library builds. set(CPM_DOWNLOAD_spdlog ON)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
  rapids_cpm_spdlog(
    FMT_OPTION "EXTERNAL_FMT_HO"
    INSTALL_EXPORT_SET rmm-exports
    BUILD_EXPORT_SET rmm-exports
    # TODO: We can't support both modes right now unfortunately because we require a static spdlog
    # for the new builds to hide the symbols properly. CPM_ARGS OPTIONS "SPDLOG_BUILD_SHARED OFF"
    # "BUILD_SHARED_LIBS OFF"
  )

endfunction()

find_and_configure_spdlog()
