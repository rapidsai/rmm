# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cdef class DeviceBuffer:
    cdef unique_ptr[device_buffer] c_obj

    # Holds a reference to the DeviceMemoryResource used for allocation.
    # Ensures the MR does not get destroyed before this DeviceBuffer. `mr` is
    # needed for deallocation
    cdef DeviceMemoryResource mr

    # Holds a reference to the stream used by the underlying `device_buffer`.
    # Ensures the stream does not get destroyed before this DeviceBuffer
    cdef Stream stream

    @staticmethod
    cdef DeviceBuffer c_from_unique_ptr(
        unique_ptr[device_buffer] ptr,
        Stream stream=*,
        DeviceMemoryResource mr=*,
    )

    @staticmethod
    cdef DeviceBuffer c_to_device(const unsigned char[::1] b,
                                  Stream stream=*) except *
    cpdef copy_to_host(self, ary=*, Stream stream=*)
    cpdef copy_from_host(self, ary, Stream stream=*)
    cpdef copy_from_device(self, cuda_ary, Stream stream=*)
    cpdef bytes tobytes(self, Stream stream=*)

    cdef size_t c_size(self) except *
    cpdef void reserve(self, size_t new_capacity, Stream stream=*) except *
    cpdef void resize(self, size_t new_size, Stream stream=*) except *
    cpdef size_t capacity(self) except *
    cdef void* c_data(self) except *

    cdef device_buffer c_release(self) except *

cpdef DeviceBuffer to_device(const unsigned char[::1] b,
                             Stream stream=*)
cpdef void copy_ptr_to_host(uintptr_t db,
                            unsigned char[::1] hb,
                            Stream stream=*) except *

cpdef void copy_host_to_ptr(const unsigned char[::1] hb,
                            uintptr_t db,
                            Stream stream=*) except *

cpdef void copy_device_to_ptr(uintptr_t d_src,
                              uintptr_t d_dst,
                              size_t count,
                              Stream stream=*) except *
