# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# cython: profile = False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from rmm._lib.lib cimport c_free, cudaStream_t
from libc.stdint cimport uintptr_t


cdef class DevicePointer:
    def __cinit__(self, ptr, stream=0):
        """
        A DevicePointer wraps a raw pointer, freeing it
        via `rmmFree()` when it (the DevicePointer) goes out of
        scope. Effectively, DevicePointer takes ownership of the
        memory pointed to by the pointer.

        Paramters
        ---------
        ptr : int
            Pointer to device memory
        stream : int, optional
            CUDA stream to use for the deallocation
        """
        self.c_ptr = <void*><uintptr_t>(ptr)
        self.c_stream = <cudaStream_t><uintptr_t>(stream)

    @property
    def ptr(self):
        return int(<uintptr_t>(self.c_ptr))

    @property
    def stream(self):
        return int(<uintptr_t>(self.c_stream))

    def __dealloc__(self):
        c_free(self.c_ptr, self.c_stream)
