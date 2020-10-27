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

# Preprocessor step to specify correct pxd file with valid symbols for specific version of CUDA. 
set(preprocess_files "gpu.pxd;")
set(supported_cuda_versions "10.1;10.2;11.0")
foreach(file_p ${preprocess_files})
  get_filename_component(pxi_file ${file_p} NAME_WLE)
  set(pxi_file "${pxi_file}.pxi")
  set(cuda_major_minor "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  if(cuda_major_minor IN_LIST supported_cuda_versions)
    file(COPY "${cuda_major_minor}/${pxi_file}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
    file(RENAME "${pxi_file}" "gpu.pxd")
  else()
    MESSAGE(FATAL_ERROR "CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is not supported.")
  endif(cuda_major_minor IN_LIST supported_cuda_versions)
endforeach(file_p ${preprocess_files})

# Set the list of Cython files to build
list(APPEND cython_sources "gpu.pyx")

# Build all of the Cython targets
foreach(cython_file ${cython_sources})
  get_filename_component(cython_module_name ${cython_file} NAME_WLE)
  add_cython_target("${cython_module_name}" "${cython_file}" CXX PY3)
  add_library(${cython_module_name} MODULE ${cython_module_name})
  target_link_libraries(${cython_module_name} rmm::rmm Python3::Module)
  target_include_directories(${cython_module_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${cython_module_name} LIBRARY DESTINATION ${cython_module_name})
endforeach(cython_file ${cython_sources})
