# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(cython_sources device_buffer.pyx lib.pyx logger.pyx memory_resource.pyx cuda_stream.pyx)
set(linked_libraries rmm::rmm)

# Build all of the Cython targets
rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}"
                                                                               CXX)

add_library(_torch_allocator SHARED _torch_allocator.cpp)
# Want the output to be called _torch_allocator.so
set_target_properties(_torch_allocator PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(_torch_allocator PRIVATE rmm::rmm)
cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
           OUTPUT_VARIABLE _torch_allocator_location)
install(TARGETS _torch_allocator DESTINATION "${_torch_allocator_location}")
