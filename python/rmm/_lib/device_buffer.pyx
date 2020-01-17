from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from rmm._lib.lib cimport (cudaError_t, cudaSuccess,
                           cudaStream_t, cudaStreamSynchronize)


cdef class DeviceBuffer:

    def __cinit__(self, *, ptr=None, size=None, stream=None):
        cdef size_t c_size
        if size is None:
            c_size = <size_t>0
        else:
            c_size = <size_t>size

        cdef cudaStream_t c_stream
        if stream is None:
            c_stream = <cudaStream_t><uintptr_t>0
        else:
            c_stream = <cudaStream_t><uintptr_t>stream

        cdef void * c_ptr
        if ptr is None:
            c_ptr = <void *>NULL
        else:
            c_ptr = <void *> <uintptr_t> ptr

        with nogil:
            if c_ptr == NULL:
                self.c_obj.reset(new device_buffer(c_size, c_stream))
            else:
                self.c_obj.reset(new device_buffer(c_ptr, c_size, c_stream))

    def __len__(self):
        return self.size

    @property
    def nbytes(self):
        return self.size

    @property
    def ptr(self):
        return int(<uintptr_t>self.c_data())

    @property
    def size(self):
        return self.c_size()

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

    cpdef bytes tobytes(self, uintptr_t stream=0):
        cdef const device_buffer* dbp = self.c_obj.get()
        cdef bytes b = PyBytes_FromStringAndSize(NULL, self.c_size())
        cdef char* p = PyBytes_AS_STRING(b)
        cdef cudaError_t err

        with nogil:
            copy_to_host(dbp[0], <void*>p, <cudaStream_t>stream)
            err = cudaStreamSynchronize(<cudaStream_t>stream)
        if err != cudaSuccess:
            raise RuntimeError(f"Stream sync failed with error: {err}")

        return b

    cdef size_t c_size(self):
        return self.c_obj.get()[0].size()

    cpdef void resize(self, size_t new_size):
        self.c_obj.get()[0].resize(new_size)

    cpdef size_t capacity(self):
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self):
        return self.c_obj.get()[0].data()
