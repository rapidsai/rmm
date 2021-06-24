# Copyright (c) 2020, NVIDIA CORPORATION.
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
from libcpp cimport bool

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.lib cimport cudaStream_t


cdef class Stream:
    cdef cudaStream_t _cuda_stream
    cdef object _owner

    @staticmethod
    cdef Stream _from_cudaStream_t(cudaStream_t s, object owner=*)

    cdef cuda_stream_view view(self) nogil except *
    cdef void c_synchronize(self) nogil except *
    cdef bool c_is_default(self) nogil except *
    cdef void _init_with_new_cuda_stream(self) except *
    cdef void _init_from_stream(self, Stream stream) except *
