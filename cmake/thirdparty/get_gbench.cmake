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

# Use CPM to find or clone gbench
function(find_and_configure_gbench VERSION)

  if(TARGET benchmark::benchmark)
    return()
  endif()

  rapids_cpm_find(
    benchmark ${VERSION}
    CPM_ARGS GITHUB_REPOSITORY google/benchmark VERSION ${VERSION}
    GIT_SHALLOW TRUE
    OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF")

endfunction()

find_and_configure_gbench(1.5.2)
