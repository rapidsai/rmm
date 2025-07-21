# Copyright (c) 2020-2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4)

# Add debug and NVTX options
add_definitions(-DRMM_NVTX)
add_definitions(-DRMM_LOGGING_LEVEL=DEBUG)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(nvtx_example)

project(
  nvtx_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

include(rapids-cmake)
rapids_cmake_build_type("Debug")

find_package(rmm REQUIRED)
get_target_property(RMM_INCLUDE_DIRS rmm::rmm INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "RMM include directories: ${RMM_INCLUDE_DIRS}")

# Configure your project here
add_executable(nvtx_oob src/nvtx_oob.cu)
target_compile_definitions(nvtx_oob PUBLIC -DRMM_NVTX -DRMM_LOGGING_LEVEL=DEBUG)
target_link_libraries(nvtx_oob PRIVATE rmm::rmm)
target_compile_features(nvtx_oob PRIVATE cxx_std_17)

# install(TARGETS nvtx_oob DESTINATION bin/examples/librmm) 