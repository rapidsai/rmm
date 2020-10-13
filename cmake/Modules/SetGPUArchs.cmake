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

set(GPU_ARCHS
    ""
    CACHE
      STRING
      "List of GPU architectures (semicolon-separated) to be compiled for. Empty string means to auto-detect the GPUs on the current system"
)

if("${GPU_ARCHS}" STREQUAL "")
  include(${RMM_SOURCE_DIR}/cmake/Modules/EvalGPUArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
  if("${GPU_ARCHS}" STREQUAL "ALL")
    # Check for embedded vs workstation architectures
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      # This is being built for Linux4Tegra or SBSA ARM64
      set(GPU_ARCHS "62")
      if((CUDAToolkit_VERSION_MAJOR EQUAL 9) OR (CUDAToolkit_VERSION_MAJOR GREATER 9))
        list(APPEND GPU_ARCHS "72")
      endif()
      if((CUDAToolkit_VERSION_MAJOR EQUAL 11) OR (CUDAToolkit_VERSION_MAJOR GREATER 11))
        list(APPEND GPU_ARCHS "75" "80")
      endif()
    else()
      # This is being built for an x86 or x86_64 architecture
      set(GPU_ARCHS "60")
      if((CUDAToolkit_VERSION_MAJOR EQUAL 9) OR (CUDAToolkit_VERSION_MAJOR GREATER 9))
        list(APPEND GPU_ARCHS "70")
      endif()
      if((CUDAToolkit_VERSION_MAJOR EQUAL 10) OR (CUDAToolkit_VERSION_MAJOR GREATER 10))
        list(APPEND GPU_ARCHS "75")
      endif()
      if((CUDAToolkit_VERSION_MAJOR EQUAL 11) OR (CUDAToolkit_VERSION_MAJOR GREATER 11))
        list(APPEND GPU_ARCHS "80")
      endif()
    endif()
  endif()
endif()
