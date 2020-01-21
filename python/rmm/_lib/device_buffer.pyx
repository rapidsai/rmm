from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from rmm._lib.lib cimport (cudaError_t, cudaSuccess,
                           cudaStream_t, cudaStreamSynchronize)


cdef class DeviceBuffer:

    def __cinit__(self, *,
                  uintptr_t ptr=0,
                  size_t size=0,
                  uintptr_t stream=0):
        cdef void* c_ptr
        cdef cudaStream_t c_stream

        with nogil:
            c_ptr = <void*>ptr
            c_stream = <cudaStream_t>stream

            if c_ptr == NULL:
                self.c_obj.reset(new device_buffer(size, c_stream))
            else:
                self.c_obj.reset(new device_buffer(c_ptr, size, c_stream))

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
        return int(self.c_size())

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
        cdef size_t s = dbp.size()
        if s == 0:
            return b""

        cdef cudaStream_t c_stream = <cudaStream_t>stream
        cdef bytes b = PyBytes_FromStringAndSize(NULL, s)
        cdef void* p = <void*>PyBytes_AS_STRING(b)
        cdef cudaError_t err
        with nogil:
            copy_to_host(dbp[0], p, c_stream)
            err = cudaStreamSynchronize(c_stream)
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
