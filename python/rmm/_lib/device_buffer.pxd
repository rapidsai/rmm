from rmm._lib.lib cimport cudaStream_t

cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_buffer:
        device_buffer()
        device_buffer(size_t size)
        device_buffer(const void* source_data, size_t size)
        device_buffer(const device_buffer& other)
        void resize(size_t new_size)
        void shrink_to_fit()
        void* data()
        size_t size()
        size_t capacity()
