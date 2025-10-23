# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp cimport bool


cdef extern from "rmm/cuda_stream_view.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_view:
        cuda_stream_view()
        cuda_stream_view(cudaStream_t)
        cudaStream_t value()
        bool is_default()
        bool is_per_thread_default()
        void synchronize() except +

    cdef bool operator==(cuda_stream_view const, cuda_stream_view const)

    const cuda_stream_view cuda_stream_default
    const cuda_stream_view cuda_stream_legacy
    const cuda_stream_view cuda_stream_per_thread
