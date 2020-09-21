include(FetchContent)

###################################################################################################
# - spdlog ----------------------------------------------------------------------------------------

CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.7.0
  GIT_SHALLOW TRUE
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
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG release-1.10.0
    VERSION 1.10.0
    GIT_SHALLOW TRUE
    OPTIONS
      "INSTALL_GTEST OFF"
    )
endif()

###################################################################################################
# - googlebenchmark -------------------------------------------------------------------------------

if (BUILD_BENCHMARKS)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.5.2
    GIT_SHALLOW TRUE
    OPTIONS
      "BENCHMARK_ENABLE_TESTING OFF"
      "BENCHMARK_ENABLE_INSTALL OFF"
      )
endif()
