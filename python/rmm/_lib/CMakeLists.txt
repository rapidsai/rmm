# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# Set the list of Cython files to build
set(cython_sources
    device_buffer
    lib
    memory_resource
    cuda_stream)

# Build all of the Cython targets
foreach(cython_module ${cython_sources})
  add_cython_target("${cython_module}" CXX PY3)
  add_library(${cython_module} MODULE ${cython_module})
  # TODO: This doesn't seem to be necessary for some reason. Are the Python
  # headers somehow being included without it?
  #python_extension_module(${cython_module})

  # To avoid libraries being prefixed with "lib".
  set_target_properties(${cython_module} PROPERTIES PREFIX "")

  target_link_libraries(${cython_module} rmm::rmm Python3::Module)
  target_include_directories(${cython_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${cython_module} DESTINATION rmm/_lib)
endforeach(cython_module ${cython_sources})
