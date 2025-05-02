# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

include(${CMAKE_CURRENT_LIST_DIR}/versions.cmake)

set(CPM_DOWNLOAD_VERSION v0.40.5)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

# find or build it via CPM
CPMFindPackage(
  NAME rmm
  FIND_PACKAGE_ARGUMENTS "PATHS ${rmm_ROOT} ${rmm_ROOT}/latest" GIT_REPOSITORY
                         https://github.com/rapidsai/rmm
  GIT_TAG ${RMM_TAG}
  GIT_SHALLOW TRUE)
