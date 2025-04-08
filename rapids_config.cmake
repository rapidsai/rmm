# =============================================================================
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

# Enforce the minimum required CMake version for all users
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _rapids_version)
if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])]])
  set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
  set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
  set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
else()
  string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
  message(
    FATAL_ERROR
      "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}")
endif()

# Allow users to control which version is used
if(NOT rapids-cmake-version)
  # Define a default version if the user doesn't set one
  set(rapids-cmake-version ${RAPIDS_VERSION_MAJOR_MINOR})
endif()

# Allow users to control which GitHub repo is fetched
if(NOT rapids-cmake-repo)
  # Define a default repo if the user doesn't set one
  set(rapids-cmake-repo rapidsai/rapids-cmake)
endif()

# Allow users to control which branch is fetched
if(NOT rapids-cmake-branch)
  # Define a default branch if the user doesn't set one
  set(rapids-cmake-branch "branch-${rapids-cmake-version}")
endif()

# Allow users to control the exact URL passed to FetchContent
if(NOT rapids-cmake-url)
  # Construct a default URL if the user doesn't set one
  set(rapids-cmake-url "https://github.com/${rapids-cmake-repo}/")

  # In order of specificity
  if(rapids-cmake-fetch-via-git)
    if(rapids-cmake-sha)
      # An exact git SHA takes precedence over anything
      set(rapids-cmake-value-to-clone "${rapids-cmake-sha}")
    elseif(rapids-cmake-tag)
      # Followed by a git tag name
      set(rapids-cmake-value-to-clone "${rapids-cmake-tag}")
    else()
      # Or if neither of the above two were defined, use a branch
      set(rapids-cmake-value-to-clone "${rapids-cmake-branch}")
    endif()
  else()
    if(rapids-cmake-sha)
      # An exact git SHA takes precedence over anything
      set(rapids-cmake-value-to-clone "archive/${rapids-cmake-sha}.zip")
    elseif(rapids-cmake-tag)
      # Followed by a git tag name
      set(rapids-cmake-value-to-clone "archive/refs/tags/${rapids-cmake-tag}.zip")
    else()
      # Or if neither of the above two were defined, use a branch
      set(rapids-cmake-value-to-clone "archive/refs/heads/${rapids-cmake-branch}.zip")
    endif()
  endif()
endif()

include(FetchContent)
if(rapids-cmake-fetch-via-git)
  FetchContent_Declare(
    rapids-cmake
    GIT_REPOSITORY "${rapids-cmake-url}"
    GIT_TAG "${rapids-cmake-value-to-clone}")
else()
  string(APPEND rapids-cmake-url "${rapids-cmake-value-to-clone}")
  FetchContent_Declare(rapids-cmake URL "${rapids-cmake-url}")
endif()
FetchContent_GetProperties(rapids-cmake)
if(rapids-cmake_POPULATED)
  # Something else has already populated rapids-cmake, only thing we need to do is setup the
  # CMAKE_MODULE_PATH
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
else()
  FetchContent_MakeAvailable(rapids-cmake)
endif()
