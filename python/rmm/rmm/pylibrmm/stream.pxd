# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm.librmm.cuda_stream cimport cuda_stream
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


@cython.final
cdef class CudaStream:
    cdef unique_ptr[cuda_stream] c_obj
    cdef cudaStream_t value(self) except * nogil
    cdef bool is_valid(self) except * nogil


cdef class Stream:
    cdef cudaStream_t _cuda_stream
    cdef object _owner

    @staticmethod
    cdef Stream _from_cudaStream_t(cudaStream_t s, object owner=*)

    cdef cuda_stream_view view(self) noexcept nogil
    cdef void c_synchronize(self) except * nogil
    cdef bool c_is_default(self) noexcept nogil
    cdef void _init_with_new_cuda_stream(self) except *
    cdef void _init_from_stream(self, Stream stream) except *
