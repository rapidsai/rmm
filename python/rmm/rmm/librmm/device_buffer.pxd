# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream_ref cimport stream_ref
from rmm.librmm.memory_resource cimport any_resource, device_accessible


cdef extern from "rmm/mr/per_device_resource.hpp" namespace "rmm" nogil:
    cdef cppclass cuda_device_id:
        ctypedef int value_type
        cuda_device_id()
        cuda_device_id(value_type id)
        value_type value()

    cdef cuda_device_id get_current_cuda_device()

cdef extern from "rmm/prefetch.hpp" namespace "rmm" nogil:
    cdef void prefetch(const void* ptr,
                       size_t bytes,
                       cuda_device_id device,
                       stream_ref stream) except +

cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_buffer:
        device_buffer()
        device_buffer(
            size_t size,
            stream_ref stream,
            any_resource[device_accessible] mr
        ) except +
        device_buffer(
            const void* source_data,
            size_t size,
            stream_ref stream,
            any_resource[device_accessible] mr
        ) except +
        device_buffer(
            const device_buffer buf,
            stream_ref stream,
            any_resource[device_accessible] mr
        ) except +
        void reserve(size_t new_capacity, stream_ref stream) except +
        void resize(size_t new_size, stream_ref stream) except +
        void shrink_to_fit(stream_ref stream) except +
        void* data()
        size_t size()
        size_t capacity()
