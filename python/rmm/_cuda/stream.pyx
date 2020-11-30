from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.lib cimport cudaStream_t
from rmm._lib.cuda_stream import CudaStream

from numba import cuda
import cupy


cdef class Stream:
    def __init__(self, obj=None):
        """
        A Stream represents a CUDA stream.

        Parameters
        ----------
        obj: optional
            * If None (the default), a new CUDA stream is created.
            * If an integer is provided, it is assumed to be a handle
              to an existing CUDA stream.
            * If a Numba or CUDA stream is provided, we make a thin
              wrapper around it.
        """
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

    cdef cuda_stream_view view(self) nogil except *:
        """
        Generate a rmm::cuda_stream_view from this Stream instance
        """
        return cuda_stream_view(<cudaStream_t><uintptr_t>(self._ptr))

    cpdef bool is_default(self) except *:
        """
        Check if we are the default CUDA stream
        """
        return self.view().is_default()

    cpdef void synchronize(self) except *:
        """
        Synchronize the CUDA stream
        """
        self.view().synchronize()

    def _from_numba_stream(self, stream):
        self._ptr = stream.handle.value
        self._owner = stream

    def _from_cupy_stream(self, stream):
        self._ptr = stream.ptr
        self._owner = stream

DEFAULT_STREAM = Stream(0)
