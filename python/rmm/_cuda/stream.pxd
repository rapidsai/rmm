from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm._lib.cuda_stream_view cimport cuda_stream_view


cdef class Stream:
    cdef public:
        uintptr_t _ptr
        object _owner

    cdef cuda_stream_view view(self) nogil except *
    cpdef bool is_default(self) except *
    cpdef void synchronize(self) except *
