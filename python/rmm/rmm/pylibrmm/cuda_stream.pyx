# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

cimport cython
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool

from rmm.librmm.cuda_stream cimport cuda_stream


@cython.final
cdef class CudaStream:
    """
    Wrapper around a CUDA stream with RAII semantics.
    When a CudaStream instance is GC'd, the underlying
    CUDA stream is destroyed.
    """
    def __cinit__(self):
        with nogil:
            self.c_obj.reset(new cuda_stream())

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    cdef cudaStream_t value(self) except * nogil:
        return self.c_obj.get()[0].value()

    cdef bool is_valid(self) except * nogil:
        return self.c_obj.get()[0].is_valid()
