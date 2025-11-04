# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uint32_t
from libcpp cimport bool

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "rmm/cuda_stream.hpp" namespace "rmm" nogil:

    cpdef enum class cuda_stream_flags "rmm::cuda_stream::flags" (uint32_t):
        sync_default "rmm::cuda_stream::flags::sync_default"
        non_blocking "rmm::cuda_stream::flags::non_blocking"
    cdef cppclass cuda_stream:
        cuda_stream() except +
        bool is_valid() except +
        cudaStream_t value() except +
        cuda_stream_view view() except +
        void synchronize() except +
        void synchronize_no_throw()
