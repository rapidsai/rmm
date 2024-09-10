# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from libcpp.pair cimport pair

from rmm.cpp.cpp_cuda_stream_view cimport cuda_stream_view


cdef extern from "rmm/mr/device/device_memory_resource.hpp" \
        namespace "rmm::mr" nogil:
    cdef cppclass device_memory_resource:
        void* allocate(size_t bytes) except +
        void* allocate(size_t bytes, cuda_stream_view stream) except +
        void deallocate(void* ptr, size_t bytes) except +
        void deallocate(
            void* ptr,
            size_t bytes,
            cuda_stream_view stream
        ) except +

cdef extern from "rmm/cuda_device.hpp" namespace "rmm" nogil:
    size_t percent_of_free_device_memory(int percent) except +
    pair[size_t, size_t] available_device_memory() except +
