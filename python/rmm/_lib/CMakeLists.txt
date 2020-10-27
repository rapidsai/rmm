#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Set the list of Cython files to build
list(APPEND cython_sources "device_buffer.pyx")
list(APPEND cython_sources "lib.pyx")
list(APPEND cython_sources "memory_resource.pyx")

# Build all of the Cython targets
foreach(cython_file ${cython_sources})
  get_filename_component(cython_module_name ${cython_file} NAME_WLE)
  add_cython_target("${cython_module_name}" "${cython_file}" CXX PY3)
  add_library(${cython_module_name} MODULE ${cython_module_name})
  target_link_libraries(${cython_module_name} rmm::rmm Python3::Module)
  target_include_directories(${cython_module_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${cython_module_name} LIBRARY DESTINATION ${cython_module_name})
endforeach(cython_file ${cython_sources})
