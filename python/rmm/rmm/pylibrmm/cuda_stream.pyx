# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
