# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_uvector[T]:
        device_uvector(size_t size, cuda_stream_view  stream) except +
        T* element_ptr(size_t index)
        void set_element(size_t element_index, const T& v, cuda_stream_view s)
        void set_element_async(
            size_t element_index,
            const T& v,
            cuda_stream_view s
        ) except +
        T front_element(cuda_stream_view s) except +
        T back_element(cuda_stream_view s) except +
        void reserve(size_t new_capacity, cuda_stream_view stream) except +
        void resize(size_t new_size, cuda_stream_view stream) except +
        void shrink_to_fit(cuda_stream_view stream) except +
        device_buffer release()
        size_t capacity()
        T* data()
        size_t size()
        device_memory_resource* memory_resource()
