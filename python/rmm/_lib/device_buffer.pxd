from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t

from rmm._lib.lib cimport cudaStream_t

cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_buffer:
        device_buffer()
        device_buffer(size_t size, cudaStream_t stream=*)
        device_buffer(const void* source_data,
                      size_t size, cudaStream_t stream=*)
        device_buffer(const device_buffer& other)
        void resize(size_t new_size)
        void shrink_to_fit()
        void* data()
        size_t size()
        size_t capacity()

    void copy_to_host(const device_buffer& db, void* hb) except *
    void copy_to_host(const device_buffer& db,
                      void* hb,
                      cudaStream_t stream) except *

cdef class DeviceBuffer:
    cdef unique_ptr[device_buffer] c_obj

    @staticmethod
    cdef DeviceBuffer c_from_unique_ptr(unique_ptr[device_buffer] ptr)

    cpdef bytes tobytes(self, uintptr_t stream=*)

    cdef size_t c_size(self)
    cpdef void resize(self, size_t new_size)
    cpdef size_t capacity(self)
    cdef void* c_data(self)


cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
