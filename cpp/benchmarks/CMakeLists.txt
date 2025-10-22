# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Build options
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)

if(PER_THREAD_DEFAULT_STREAM)
  message(STATUS "RMM: Building benchmarks with per-thread default stream")
endif()

# compiler function

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench BENCH_NAME)
  add_executable(${BENCH_NAME} ${ARGN}
                               "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
  target_include_directories(${BENCH_NAME} PRIVATE "$<BUILD_INTERFACE:${RMM_SOURCE_DIR}>")
  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RMM_BINARY_DIR}/gbenchmarks>"
               CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
               INSTALL_RPATH "\$ORIGIN/../../../lib")
  target_link_libraries(${BENCH_NAME} PRIVATE benchmark::benchmark pthread rmm)
  target_compile_definitions(
    ${BENCH_NAME} PUBLIC "RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${RMM_LOGGING_LEVEL}")

  if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(${BENCH_NAME} PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif()

  target_compile_options(${BENCH_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
                                              -Wno-unknown-pragmas>)

  if(DISABLE_DEPRECATION_WARNING)
    target_compile_options(
      ${BENCH_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
    target_compile_options(${BENCH_NAME}
                           PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
  endif()

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/librmm
    EXCLUDE_FROM_ALL)
endfunction(ConfigureBench)

# random allocations benchmark
ConfigureBench(RANDOM_ALLOCATIONS_BENCH random_allocations/random_allocations.cpp)

# replay benchmark
ConfigureBench(REPLAY_BENCH replay/replay.cpp)

# uvector benchmark
ConfigureBench(UVECTOR_BENCH device_uvector/device_uvector_bench.cu)

# cuda_stream_pool benchmark
ConfigureBench(CUDA_STREAM_POOL_BENCH cuda_stream_pool/cuda_stream_pool_bench.cpp)

# multi stream allocations
ConfigureBench(MULTI_STREAM_ALLOCATIONS_BENCH
               multi_stream_allocations/multi_stream_allocations_bench.cu)

# async priming benchmark
ConfigureBench(ASYNC_PRIMING_BENCH async_priming/async_priming_bench.cpp)
