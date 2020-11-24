from libc.stdint cimport uintptr_t

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.lib cimport cudaStream_t
from rmm._lib.cuda_stream import CudaStream

from numba import cuda
import cupy


cdef class Stream:
    def __init__(self, stream=None):
        if isinstance(stream, cuda.cudadrv.driver.Stream):
            self._from_numba_stream(stream)
        elif isinstance(stream, cupy.cuda.stream.Stream):
            self._from_cupy_stream(stream)
        else:
            stream = CudaStream()
            self._ptr = stream.value()
            self._owner = stream

    def _from_numba_stream(self, stream):
        self._ptr = stream.handle.value
        self._owner = stream

    def _from_cupy_stream(self, stream):
        self._ptr = stream.ptr
        self._owner = stream

    cdef cuda_stream_view view(self) except *:
        return cuda_stream_view(<cudaStream_t><uintptr_t>(self._ptr))
