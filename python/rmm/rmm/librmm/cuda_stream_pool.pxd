# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from libc.stdint cimport uint32_t

cdef extern from "rmm/cuda_stream.hpp" namespace "rmm" nogil:
    cpdef enum class cuda_stream_flags "rmm::cuda_stream::flags" (uint32_t):
        sync_default "rmm::cuda_stream::flags::sync_default"
        non_blocking "rmm::cuda_stream::flags::non_blocking"

cdef extern from "rmm/cuda_stream_pool.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_pool:
        cuda_stream_pool(size_t pool_size)
        cuda_stream_pool(size_t pool_size, cuda_stream_flags flag)
        cuda_stream_view get_stream()
        cuda_stream_view get_stream(size_t stream_id) except +
        size_t get_pool_size()
