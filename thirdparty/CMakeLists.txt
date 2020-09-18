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

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.7.0
    GIT_SHALLOW    true
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

message(STATUS "Fetching Thrust")

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    # August 28, 2020
    GIT_TAG        52a8bda46c5c2128414d1d47f546b486ff0be2f0
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(thrust)
  # We are not using the thrust CMake targets, so no need to call `add_subdirectory()`.
endif()
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
