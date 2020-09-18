#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RMM_BENCHS LANGUAGES C CXX CUDA)

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY CUDA::cudart_static)
else()
    set(CUDART_LIBRARY CUDA::cudart)
endif(CUDA_STATIC_RUNTIME)

if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
endif(USE_NVTX)

###################################################################################################
# - compiler function

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${CMAKE_BENCH_NAME} PRIVATE
                               "${GBENCH_INCLUDE_DIR}"
                               "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                               "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
                               "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
                               "$<INSTALL_INTERFACE:include>")
    target_link_directories(${CMAKE_BENCH_NAME} PRIVATE "${GBENCH_LIBRARY_DIR}")
    
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark pthread
                          spdlog::spdlog_header_only ${CUDART_LIBRARY})

    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    
    target_compile_definitions(${CMAKE_BENCH_NAME} PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGGING_LEVEL})
    
    if(USE_NVTX)
      target_compile_definitions(${CMAKE_BENCH_NAME} PUBLIC NVTX_DISABLE)
    endif(USE_NVTX)
endfunction(ConfigureBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - random allocations benchmark

set(RANDOM_ALLOCATIONS_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/random_allocations/random_allocations.cpp")

ConfigureBench(RANDOM_ALLOCATIONS_BENCH "${RANDOM_ALLOCATIONS_BENCH_SRC}")
###################################################################################################
# - replay benchmark

set(REPLAY_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/replay/replay.cpp")

ConfigureBench(REPLAY_BENCH "${REPLAY_BENCH_SRC}")
###################################################################################################
# - uvector benchmark

set(UVECTOR_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector/device_uvector_bench.cu")

ConfigureBench(UVECTOR_BENCH "${UVECTOR_BENCH_SRC}")
