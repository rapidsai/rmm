from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.lib cimport cudaStream_t
from rmm._lib.cuda_stream import CudaStream

from numba import cuda
import cupy


cdef class Stream:
    def __init__(self, obj=None):
        if isinstance(obj, cuda.cudadrv.driver.Stream):
            self._from_numba_stream(obj)
        elif isinstance(obj, cupy.cuda.stream.Stream):
            self._from_cupy_stream(obj)
        elif obj is None:
            stream = CudaStream()
            self._ptr = obj.value()
            self._owner = obj
        elif isinstance(obj, Stream):
            self._ptr, self._owner = obj._ptr, obj._owner
        elif isinstance(obj, int):
            self._ptr = obj
            self._owner = None
        else:
            raise TypeError("obj must be None or stream")

    def _from_numba_stream(self, stream):
        self._ptr = stream.handle.value
        self._owner = stream

    def _from_cupy_stream(self, stream):
        self._ptr = stream.ptr
        self._owner = stream

    cdef cuda_stream_view view(self) nogil except *:
        return cuda_stream_view(<cudaStream_t><uintptr_t>(self._ptr))

    cpdef bool is_default(self) except *:
        return self.view().is_default()

    cpdef void synchronize(self) except *:
        self.view().synchronize()

DEFAULT_STREAM = Stream(0)
