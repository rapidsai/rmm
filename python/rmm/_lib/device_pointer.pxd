from rmm._lib.lib cimport c_free, cudaStream_t

cdef class DevicePointer:
    cdef void* c_ptr
    cdef cudaStream_t c_stream
