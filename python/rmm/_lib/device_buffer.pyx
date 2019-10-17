import functools
import operator

from libc.stdint cimport uintptr_t

import numpy as np

from rmm._lib.device_buffer cimport *


cdef device_buffer buffer_from_array_interface(desc):
    """
    Construct a device_buffer from an array_interface
    """
    ptr = <void*> <uintptr_t>desc['data'][0]
    itemsize = int(desc['typestr'][2:])
    size = functools.reduce(operator.mul, desc['shape'])
    return device_buffer(ptr, size * itemsize)


cdef class DeviceBuffer:
    cdef device_buffer c_obj

    def __cinit__(self, data=None, size=None):
        """
        Construct a DeviceBuffer.

        Parameters
        ----------
        data : array_like, DeviceBuffer, or None
            If array_like, constructs a DeviceBuffer from the pointer
            to the underlying host or device memory. If None, *size*
            must be specified.
        size : int
            Size (in bytes) of the memory allocation required.
        """
        if isinstance(data, memoryview):
            self.c_obj = buffer_from_array_interface(np.array(data).__array_interface__)
        if hasattr(data, "__array_interface__"):
            self.c_obj = buffer_from_array_interface(data.__array_interface__)
        elif hasattr(data, "__cuda_array_interface__"):
            self.c_obj = buffer_from_array_interface(data.__cuda_array_interface__)
        elif isinstance(data, DeviceBuffer):
            self.c_obj = device_buffer(<device_buffer&> data.c_obj)
        elif data is None:
            if size is None:
                raise ValueError(f"Either data or size is required")
            self.c_obj = device_buffer(<size_t> size)
        else:
            raise TypeError(f"Can't create DeviceBuffer from {type(data).__name__}")

    cdef void* data(self):
        return self.c_obj.data()
    
    cpdef void resize(self, size_t new_size):
        self.c_obj.resize(new_size)

    cpdef size_t size(self):
        return self.c_obj.size()
    
    cpdef size_t capacity(self):
        return self.c_obj.capacity()
    
