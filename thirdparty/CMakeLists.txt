include(FetchContent)

###################################################################################################
# - cnmem -----------------------------------------------------------------------------------------

FetchContent_Declare(
    cnmem
    GIT_REPOSITORY https://github.com/NVIDIA/cnmem.git
    GIT_TAG        37896cc9bfc6536a8c878a1e675835c22d827821
    GIT_SHALLOW    true
)

FetchContent_GetProperties(cnmem)
if(NOT cnmem_POPULATED)
  FetchContent_Populate(cnmem)
  # We are not using the cnmem CMake targets, so no need to call `add_subdirectory()`.
endif()
set(CNMEM_INCLUDE_DIR "${cnmem_SOURCE_DIR}/include" PARENT_SCOPE)
set(CNMEM_SOURCE_DIR "${cnmem_SOURCE_DIR}/src" PARENT_SCOPE)

###################################################################################################
# - spdlog ----------------------------------------------------------------------------------------

CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.7.0
  )

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

message(STATUS "Fetching Thrust")

CPMAddPackage(
  NAME thrust
  GITHUB_REPOSITORY thrust/thrust
  # August 28, 2020
  GIT_TAG 52a8bda46c5c2128414d1d47f546b486ff0be2f0
  VERSION 1.9.10mod
  DOWNLOAD_ONLY TRUE
)

set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}" PARENT_SCOPE)

###################################################################################################
# - googletest ------------------------------------------------------------------------------------

if (BUILD_TESTS)
  message(STATUS "Fetching GoogleTest")

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           release-1.8.0
    GIT_SHALLOW       true
    )

  FetchContent_GetProperties(googletest)
  if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    set(CMAKE_POLICY_DEFAULT_CMP0048 NEW) # suppress warning because googletest doesn't specify its version
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

###################################################################################################
# - googlebenchmark -------------------------------------------------------------------------------

if (BUILD_BENCHMARKS)
  message(STATUS "Fetching GoogleBenchmark")

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY    https://github.com/google/benchmark.git
    GIT_TAG           v1.5.1
    GIT_SHALLOW       true
    )

  FetchContent_GetProperties(googlebenchmark)
  if (NOT googlebenchmark_POPULATED)
    FetchContent_Populate(googlebenchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()
