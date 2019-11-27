import cython
import functools
import operator

from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t

import numpy as np

cdef class DeviceBuffer:

    def __cinit__(self, ptr=None, size=None):
        if size is None:
            size = 0

        cdef void * data
        if ptr is None:
            self.c_obj.reset(new device_buffer(<size_t>size))
        else:
            data = <void *> <uintptr_t> ptr
            self.c_obj.reset(new device_buffer(data, size))

    def __init__(self, ptr=None, size=None):
        pass

    def __len__(self):
        return self.size

    @property
    def nbytes(self):
        return self.size

    @property
    def ptr(self):
        return int(<uintptr_t>self.c_obj.get()[0].data())

    @property
    def size(self):
        return self.c_obj.get()[0].size()

    @property
    def __cuda_array_interface__(self):
        cdef dict intf = {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": (1,),
            "typestr": "|u1",
            "version": 0
        }
        return intf

    @staticmethod
    cdef DeviceBuffer c_from_unique_ptr(unique_ptr[device_buffer] ptr):
        cdef DeviceBuffer buf = DeviceBuffer.__new__(DeviceBuffer)
        buf.c_obj = move(ptr)
        return buf

    cdef size_t c_size(self):
        return self.c_obj.get()[0].size()

    cpdef void resize(self, size_t new_size):
        self.c_obj.get()[0].resize(new_size)

    cpdef size_t capacity(self):
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self):
        return self.c_obj.get()[0].data()
