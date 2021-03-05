# Copyright (c) 2021, NVIDIA CORPORATION.

from rmm._lib.device_buffer cimport device_buffer
from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.memory_resource cimport device_memory_resource


cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_uvector[T]:
        device_uvector(size_t size, cuda_stream_view  stream) except +
        T* element_ptr(size_t index)
        void set_element(size_t element_index, const T& v, cuda_stream_view s)
        void set_element_async(
            size_t element_index,
            const T& v,
            cuda_stream_view s
        ) except +
        T front_element(cuda_stream_view s) except +
        T back_element(cuda_stream_view s) except +
        void resize(size_t new_size, cuda_stream_view stream) except +
        void shrink_to_fit(cuda_stream_view stream) except +
        device_buffer release()
        size_t capacity()
        T* data()
        size_t size()
        device_memory_resource* memory_resource()
