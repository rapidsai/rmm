# =============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

# Fetch Google Benchmark

CPMFindPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.2
  GIT_SHALLOW TRUE
  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF")

# Auto-detect available GPU compute architectures

include(${RMM_SOURCE_DIR}/cmake/Modules/SetGPUArchs.cmake)
message(STATUS "RMM: Building benchmarks with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Build options

enable_language(CUDA)

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)

if(PER_THREAD_DEFAULT_STREAM)
  message(STATUS "RMM: Building benchmarks with per-thread default stream")
endif(PER_THREAD_DEFAULT_STREAM)

# compiler function

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench BENCH_NAME BENCH_SRC)
  add_executable(${BENCH_NAME} ${BENCH_SRC}
                               "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
  target_include_directories(${BENCH_NAME} PRIVATE "$<BUILD_INTERFACE:${RMM_SOURCE_DIR}>")
  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RMM_BINARY_DIR}/gbenchmarks>"
               CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_link_libraries(${BENCH_NAME} benchmark::benchmark pthread rmm)
  target_compile_definitions(${BENCH_NAME}
                             PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

  if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(${BENCH_NAME} PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif(PER_THREAD_DEFAULT_STREAM)

  target_compile_options(${BENCH_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
                                              -Wno-error=deprecated-declarations>)
  if(DISABLE_DEPRECATION_WARNING)
    target_compile_options(
      ${BENCH_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
    target_compile_options(${BENCH_NAME}
                           PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
  endif(DISABLE_DEPRECATION_WARNING)

endfunction(ConfigureBench)

# benchmark sources

# random allocations benchmark

set(RANDOM_ALLOCATIONS_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/random_allocations/random_allocations.cpp")

ConfigureBench(RANDOM_ALLOCATIONS_BENCH "${RANDOM_ALLOCATIONS_BENCH_SRC}")

# replay benchmark

set(REPLAY_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/replay/replay.cpp")

ConfigureBench(REPLAY_BENCH "${REPLAY_BENCH_SRC}")

# uvector benchmark

set(UVECTOR_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector/device_uvector_bench.cu")

ConfigureBench(UVECTOR_BENCH "${UVECTOR_BENCH_SRC}")

# cuda_stream_pool benchmark

set(CUDA_STREAM_POOL_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_pool/cuda_stream_pool_bench.cpp")

ConfigureBench(CUDA_STREAM_POOL_BENCH "${CUDA_STREAM_POOL_BENCH_SRC}")
