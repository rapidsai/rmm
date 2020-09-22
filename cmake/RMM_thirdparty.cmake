include(FetchContent)

###################################################################################################
# - spdlog ----------------------------------------------------------------------------------------

CPMFindPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.7.0
  GIT_SHALLOW TRUE
  )

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

CPMAddPackage(
  NAME thrust
  GITHUB_REPOSITORY thrust/thrust
  GIT_TAG 1.10.0
  VERSION 1.10.0
  GIT_SHALLOW TRUE
  DOWNLOAD_ONLY TRUE
)

set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")

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
