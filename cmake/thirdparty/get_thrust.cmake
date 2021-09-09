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

# Use CPM to find or clone thrust
function(find_and_configure_thrust)

  include(${rapids-cmake-dir}/cpm/thrust.cmake)
  rapids_cpm_thrust(NAMESPACE rmm)

  # We don't list the export set information in rapids_cpm_thrust as we don't want to install Thrust
  # as part of rmm install process. Doing so would stop consumers such as cudf from using patched
  # versions, which they require for improved build times.
  rapids_export_package(BUILD Thrust rmm-exports)
  rapids_export_package(INSTALL Thrust rmm-exports)

endfunction()

find_and_configure_thrust()
