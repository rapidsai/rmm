from libcpp.memory cimport unique_ptr
from rmm._lib.lib cimport cudaStream_t


cdef extern from "rmm/device_pointer.hpp" namespace "rmm" nogil:
    cdef cppclass device_pointer:
        device_pointer(const void* ptr)
        device_pointer(const void* ptr, cudaStream_t stream)
        void* ptr()
        cudaStream_t stream()

cdef class DevicePointer:
    cdef unique_ptr[device_pointer] c_obj
    cdef void* c_ptr(self)
    cdef cudaStream_t c_stream(self)
