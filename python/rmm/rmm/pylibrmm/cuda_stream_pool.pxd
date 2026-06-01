# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool


@cython.final
cdef class CudaStreamPool:
    cdef shared_ptr[cuda_stream_pool] c_obj
