import functools
import operator

from libc.stdint cimport uintptr_t

import numpy as np


cdef class DeviceBuffer:

    def __cinit__(self, ptr=None, size=None):
        cdef void * data
        if ptr is None:
            data = NULL
        else:
            data = <void *> <uintptr_t> ptr
        if size is None:
            size = 0
        self.c_obj = new device_buffer(data, size)
        
    @property
    def ptr(self):
        return int(<uintptr_t>self.c_obj[0].data())

    @property
    def size(self):
        return self.c_obj[0].size()

    @staticmethod
    cdef DeviceBuffer from_ptr(device_buffer *ptr):
        cdef DeviceBuffer buf = DeviceBuffer.__new__(DeviceBuffer)
        buf.c_obj = ptr
        return buf
    
    cpdef size_t size(self):
        return self.c_obj[0].size()
    
    cpdef void resize(self, size_t new_size):
        self.c_obj[0].resize(new_size)
    
    cpdef size_t capacity(self):
        return self.c_obj[0].capacity()

    cdef void* data(self):
        return self.c_obj[0].data()
    
    def __dealloc__(self):
        del self.c_obj
