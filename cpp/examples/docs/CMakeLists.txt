# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4)

include(../set_cuda_architecture.cmake)

# initialize CUDA architectures
rapids_cuda_init_architectures(docs_examples)

project(
  docs_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# One executable per user guide page
set(DOCS_CUDA_EXAMPLES guide stream_ordered_allocation managed_memory)
set(DOCS_CXX_EXAMPLES introduction choosing_memory_resources logging installation)

foreach(example ${DOCS_CUDA_EXAMPLES})
  add_executable(docs_${example} src/${example}.cu)
  target_link_libraries(docs_${example} PRIVATE rmm::rmm)
  target_compile_features(docs_${example} PRIVATE cxx_std_17)
  install(TARGETS docs_${example} DESTINATION bin/examples/librmm)
endforeach()

foreach(example ${DOCS_CXX_EXAMPLES})
  add_executable(docs_${example} src/${example}.cpp)
  target_link_libraries(docs_${example} PRIVATE rmm::rmm)
  target_compile_features(docs_${example} PRIVATE cxx_std_17)
  install(TARGETS docs_${example} DESTINATION bin/examples/librmm)
endforeach()
