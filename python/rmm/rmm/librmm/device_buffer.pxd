# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "rmm/mr/device/per_device_resource.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_device_id:
        ctypedef int value_type
        cuda_device_id()
        cuda_device_id(value_type id)
        value_type value()

    cdef cuda_device_id get_current_cuda_device()

cdef extern from "rmm/prefetch.hpp" namespace "rmm" nogil:
    cdef void prefetch(const void* ptr,
                       size_t bytes,
                       cuda_device_id device,
                       cuda_stream_view stream) except +

cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_buffer:
        device_buffer()
        device_buffer(
            size_t size,
            cuda_stream_view stream,
            device_memory_resource *
        ) except +
        device_buffer(
            const void* source_data,
            size_t size,
            cuda_stream_view stream,
            device_memory_resource *
        ) except +
        device_buffer(
            const device_buffer buf,
            cuda_stream_view stream,
            device_memory_resource *
        ) except +
        void reserve(size_t new_capacity, cuda_stream_view stream) except +
        void resize(size_t new_size, cuda_stream_view stream) except +
        void shrink_to_fit(cuda_stream_view stream) except +
        void* data()
        size_t size()
        size_t capacity()
