cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RMM_BENCHS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark pthread rmm )
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
endfunction(ConfigureBench)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GTEST_INCLUDE_DIR}"
                    "${GBENCH_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - random allocations benchmark --------------------------------------------------------------------------------

set(RANDOM_ALLOCATIONS_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/random_allocations/random_allocations.cpp")

ConfigureBench(RANDOM_ALLOCATIONS_BENCH "${RANDOM_ALLOCATIONS_BENCH_SRC}")
###################################################################################################
# - replay benchmark --------------------------------------------------------------------------------

set(REPLAY_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/replay/replay.cpp")

ConfigureBench(REPLAY_BENCH "${REPLAY_BENCH_SRC}")
###################################################################################################
# - uvector benchmark --------------------------------------------------------------------------------

set(UVECTOR_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector/device_uvector_bench.cu")

ConfigureBench(UVECTOR_BENCH "${UVECTOR_BENCH_SRC}")
