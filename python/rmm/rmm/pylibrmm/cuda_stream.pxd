# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm.librmm.cuda_stream cimport cuda_stream


@cython.final
cdef class CudaStream:
    cdef unique_ptr[cuda_stream] c_obj
    cdef cudaStream_t value(self) except * nogil
    cdef bool is_valid(self) except * nogil
