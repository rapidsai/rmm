# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4)

include(../set_cuda_architecture.cmake)

# initialize CUDA architectures
rapids_cuda_init_architectures(ipc_pool_sharing)

project(
  ipc_pool_sharing
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# Pool exporter executable
add_executable(pool_exporter src/pool_exporter.cu)
target_link_libraries(pool_exporter PRIVATE rmm::rmm)
target_compile_features(pool_exporter PRIVATE cxx_std_20)

# Pool importer executable
add_executable(pool_importer src/pool_importer.cu)
target_link_libraries(pool_importer PRIVATE rmm::rmm)
target_compile_features(pool_importer PRIVATE cxx_std_20)

install(TARGETS pool_exporter pool_importer DESTINATION bin/examples/librmm)
