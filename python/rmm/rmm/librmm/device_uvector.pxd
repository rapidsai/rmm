# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t

from rmm.librmm.cuda_stream_ref cimport stream_ref
from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_uvector[T]:
        device_uvector(size_t size, stream_ref stream) except +
        device_uvector(size_t size, cudaStream_t stream) except +
        T* element_ptr(size_t index)
        void set_element(size_t element_index, const T& v, stream_ref s)
        void set_element_async(
            size_t element_index,
            const T& v,
            stream_ref s
        ) except +
        T front_element(stream_ref s) except +
        T back_element(stream_ref s) except +
        void reserve(size_t new_capacity, stream_ref stream) except +
        void resize(size_t new_size, stream_ref stream) except +
        void shrink_to_fit(stream_ref stream) except +
        device_buffer release()
        size_t capacity()
        T* data()
        size_t size()
