# Copyright (c) 2024, NVIDIA CORPORATION.
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

from cuda.ccudart cimport cudaStream_t


cdef extern from "cuda/memory_resource" namespace "cuda" nogil:
    cdef cppclass stream_ref:
        stream_ref() except +
        stream_ref(cudaStream_t stream) except +

cdef extern from "rmm/aligned.hpp" namespace "rmm" nogil:
    cdef size_t CUDA_ALLOCATION_ALIGNMENT

cdef extern from "rmm/resource_ref.hpp" namespace "rmm" nogil:
    cdef cppclass device_async_resource_ref:
        void* allocate(size_t bytes, size_t alignment) except +
        void deallocate(void* ptr, size_t bytes, size_t alignment) except +
        void* allocate_async(
            size_t bytes,
            size_t alignment,
            stream_ref stream) except +
        void deallocate_async(
            void* ptr,
            size_t bytes,
            size_t alignment,
            stream_ref stream
        ) except +
