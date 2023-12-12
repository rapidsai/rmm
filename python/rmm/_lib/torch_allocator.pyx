# Copyright (c) 2023, NVIDIA CORPORATION.
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

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.memory_resource cimport device_memory_resource
from rmm._lib.per_device_resource cimport (
    cuda_device_id,
    get_per_device_resource,
)


cdef public void* allocate(
    size_t size, int device, void* stream
) except * with gil:
    cdef cuda_device_id* device_id
    cdef device_memory_resource* mr
    # Workaround for cuda_device_id not having a nullary constructor
    # and therefore not being stack allocatable in Cython code.
    device_id = new cuda_device_id(device)
    try:
        mr = get_per_device_resource(device_id[0])
    finally:
        del device_id
    cdef cuda_stream_view stream_view = cuda_stream_view(
        <cudaStream_t>(stream)
    )
    return mr[0].allocate(size, stream_view)

cdef public void deallocate(
    void* ptr, size_t size, int device, void* stream
) except * with gil:
    cdef cuda_device_id* device_id
    cdef device_memory_resource* mr
    device_id = new cuda_device_id(device)
    try:
        mr = get_per_device_resource(device_id[0])
    finally:
        del device_id
    cdef cuda_stream_view stream_view = cuda_stream_view(
        <cudaStream_t>(stream)
    )
    mr[0].deallocate(ptr, size, stream_view)
