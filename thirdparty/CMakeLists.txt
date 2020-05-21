include(FetchContent)

###################################################################################################
# - cnmem -----------------------------------------------------------------------------------------

FetchContent_Declare(
    cnmem
    GIT_REPOSITORY https://github.com/NVIDIA/cnmem
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
    GIT_TAG        v1.6.0
    GIT_SHALLOW    true
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
