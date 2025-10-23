# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


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
