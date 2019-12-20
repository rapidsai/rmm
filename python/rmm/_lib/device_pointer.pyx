from rmm._lib.lib cimport c_free, cudaStream_t
from libc.stdint cimport uintptr_t


cdef class DevicePointer:
    def __cinit__(self, ptr, stream=0):
        """
        A DevicePointer wraps a raw pointer, freeing it
        via `rmmFree()` when it (the DevicePointer) goes out of
        scope. Effectively, DevicePointer takes ownership of the
        memory pointed to by the pointer.

        Paramters
        ---------
        ptr : int
            Pointer to device memory
        stream : int, optional
            CUDA stream to use for the deallocation
        """
        cdef void* c_ptr = <void*><uintptr_t>(ptr)
        cdef cudaStream_t c_stream = <cudaStream_t><uintptr_t>(stream)
        self.c_obj.reset(new device_pointer(c_ptr, c_stream))

    cdef void* c_ptr(self):
        return self.c_obj.get()[0].ptr()

    cdef cudaStream_t c_stream(self):
        return self.c_obj.get()[0].stream()

    @property
    def ptr(self):
        return int(<uintptr_t>(self.c_ptr()))

    @property
    def stream(self):
        return int(<uintptr_t>(self.c_stream()))
