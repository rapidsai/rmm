from rmm._lib.lib cimport c_free, cudaStream_t
from libc.stdint cimport uintptr_t

cdef class DevicePointer:
    def __cinit__(self, ptr, stream=0):
        self.c_ptr = <void*><uintptr_t>(ptr)
        self.c_stream = <cudaStream_t><size_t>(stream)

    def __init__(self, ptr, stream=0):
        self.ptr = ptr
        self.stream = stream

    def __dealloc__(self):
        c_free(self.c_ptr, self.c_stream)
