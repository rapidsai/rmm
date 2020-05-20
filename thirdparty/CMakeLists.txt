include(FetchContent)

FetchContent_Declare(
    cnmem
    GIT_REPOSITORY https://github.com/NVIDIA/cnmem
    GIT_TAG        37896cc9bfc6536a8c878a1e675835c22d827821
    GIT_SHALLOW    true
)
if(NOT cnmem_POPULATED)
  FetchContent_Populate(cnmem)
  # We are not using the cnmem CMake targets, so no need to call `add_subdirectory()`.
endif()

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.6.0
    GIT_SHALLOW    true
)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  # We are not using the spdlog CMake targets, so no need to call `add_subdirectory()`.
endif()
