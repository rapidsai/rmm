from libc.stdint cimport uintptr_t

from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef class Stream:
    cdef public:
        uintptr_t _ptr
        object _owner

    cdef cuda_stream_view view(self) except *
