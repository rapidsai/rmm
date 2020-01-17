from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from rmm._lib.lib cimport (cudaError_t, cudaSuccess,
                           cudaHostAllocDefault,
                           cudaHostAlloc, cudaFreeHost,
                           cudaMemcpy, cudaMemcpyHostToHost,
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
        cdef size_t s = dbp.size()
        if s == 0:
            return b""

        cdef bytes b = PyBytes_FromStringAndSize(NULL, s)
        cdef char* bp = PyBytes_AS_STRING(b)
        cdef void *hp = NULL
        cdef cudaError_t alloc_err, memcpy_err, stream_err, free_err
        with nogil:
            alloc_err = cudaHostAlloc(&hp, s, cudaHostAllocDefault)
            if alloc_err == cudaSuccess:
                copy_to_host(dbp[0], hp, <cudaStream_t>stream)
                stream_err = cudaStreamSynchronize(<cudaStream_t>stream)
                if stream_err == cudaSuccess:
                    memcpy_err = cudaMemcpy(bp, hp, s, cudaMemcpyHostToHost)
                free_err = cudaFreeHost(hp)
        if alloc_err != cudaSuccess:
            raise RuntimeError(f"Host alloc failed with error: {alloc_err}")
        if stream_err != cudaSuccess:
            raise RuntimeError(f"Stream sync failed with error: {stream_err}")
        if memcpy_err != cudaSuccess:
            raise RuntimeError(f"Memcpy failed with error: {memcpy_err}")
        if free_err != cudaSuccess:
            raise RuntimeError(f"Host free failed with error: {free_err}")

        return b

    cdef size_t c_size(self):
        return self.c_obj.get()[0].size()

    cpdef void resize(self, size_t new_size):
        self.c_obj.get()[0].resize(new_size)

    cpdef size_t capacity(self):
        return self.c_obj.get()[0].capacity()

    cdef void* c_data(self):
        return self.c_obj.get()[0].data()
