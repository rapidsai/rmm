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
        self.c_ptr = <void*><uintptr_t>(ptr)
        self.c_stream = <cudaStream_t><uintptr_t>(stream)

    @property
    def ptr(self):
        return int(<uintptr_t>(self.c_ptr))

    @property
    def stream(self):
        return int(<uintptr_t>(self.c_stream))

    def __dealloc__(self):
        c_free(self.c_ptr, self.c_stream)
