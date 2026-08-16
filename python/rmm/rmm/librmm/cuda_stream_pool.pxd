# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream cimport cuda_stream_flags
from rmm.librmm.cuda_stream_view cimport stream_ref


cdef extern from "rmm/cuda_stream_pool.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_stream_pool:
        cuda_stream_pool(size_t pool_size)
        cuda_stream_pool(size_t pool_size, cuda_stream_flags flags)
        stream_ref get_stream()
        stream_ref get_stream(size_t stream_id) except +
        size_t get_pool_size()
