# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "rmm/cuda_stream.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream:
        cuda_stream() except +
        bool is_valid() except +
        cudaStream_t value() except +
        cuda_stream_view view() except +
        void synchronize() except +
        void synchronize_no_throw()
