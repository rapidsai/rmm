# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t


cdef extern from "<cuda/stream>" namespace "cuda" nogil:
    cdef cppclass stream_ref:
        stream_ref()
        stream_ref(cudaStream_t)
        cudaStream_t get()
        void wait() except +
