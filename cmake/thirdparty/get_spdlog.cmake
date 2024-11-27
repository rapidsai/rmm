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

# Use CPM to find or clone speedlog.
function(find_and_configure_spdlog)

  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  rapids_cpm_spdlog(
    # The conda package for fmt is hard-coded to assume that we use a preexisting fmt library. This
    # is why we have always had a libfmt linkage despite choosing to specify the header-only version
    # of fmt. We need a more robust way of modifying this to support fully self-contained build and
    # usage even in environments where fmt and/or spdlog are already present. The crudest solution
    # would be to modify the interface compile definitions and link libraries of the spdlog target,
    # if necessary. For now I'm specifying EXTERNAL_FMT_HO here so that in environments where spdlog
    # is cloned and built from source we wind up with the behavior that we expect, but we'll have to
    # resolve this properly eventually.
    FMT_OPTION "EXTERNAL_FMT_HO"
    INSTALL_EXPORT_SET rmm-exports
    BUILD_EXPORT_SET rmm-exports)
endfunction()

find_and_configure_spdlog()
