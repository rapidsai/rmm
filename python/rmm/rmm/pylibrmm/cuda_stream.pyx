# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from enum import IntEnum
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool

from rmm.librmm.cuda_stream cimport cuda_stream, cuda_stream_flags


class CudaStreamFlags(IntEnum):
    """
    Enumeration of CUDA stream creation flags.

    Attributes
    ----------
    SYNC_DEFAULT : int
        Created stream synchronizes with the default stream.
    NON_BLOCKING : int
        Created stream does not synchronize with the default stream.
    """
    SYNC_DEFAULT = <int>cuda_stream_flags.sync_default
    NON_BLOCKING = <int>cuda_stream_flags.non_blocking


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
