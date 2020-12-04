# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm._lib.cuda_stream_view cimport (
    cuda_stream_default,
    cuda_stream_legacy,
    cuda_stream_per_thread,
    cuda_stream_view,
)
from rmm._lib.lib cimport cudaStream_t

from numba import cuda

from rmm._lib.cuda_stream cimport CudaStream

from rmm._lib.cuda_stream import CudaStream


cdef class Stream:
    def __init__(self, obj=None):
        """
        A Stream represents a CUDA stream.

        Parameters
        ----------
        obj: optional
            * If None (the default), a new CUDA stream is created.
            * If a Numba or CuPy stream is provided, we make a thin
              wrapper around it.
        """
        if obj is None:
            self._init_with_new_cuda_stream()
        elif isinstance(obj, Stream):
            self._init_from_stream(obj)
        elif isinstance(obj, cuda.cudadrv.driver.Stream):
            self._init_from_numba_stream(obj)
        else:
            self._init_from_cupy_stream(obj)

    @staticmethod
    cdef Stream _from_cudaStream_t(cudaStream_t s, object owner=None):
        """
        Construct a Stream from a cudaStream_t.
        """
        cdef Stream obj = Stream.__new__(Stream)
        obj._cuda_stream = s
        obj._owner = owner
        return obj

    cdef cuda_stream_view view(self) nogil except *:
        """
        Generate a rmm::cuda_stream_view from this Stream instance
        """
        return cuda_stream_view(<cudaStream_t><uintptr_t>(self._cuda_stream))

    cdef void c_synchronize(self) nogil except *:
        """
        Synchronize the CUDA stream.
        This function *must* be called in a `with nogil` block
        """
        self.view().synchronize()

    def synchronize(self):
        """
        Synchronize the CUDA stream
        """
        with nogil:
            self.c_synchronize()

    cdef bool c_is_default(self) nogil except *:
        """
        Check if we are the default CUDA stream
        """
        return self.view().is_default()

    def is_default(self):
        """
        Check if we are the default CUDA stream
        """
        return self.c_is_default()

    def _init_from_numba_stream(self, obj):
        self._cuda_stream = <cudaStream_t>(obj.handle.value)
        self._owner = obj

    def _init_from_cupy_stream(self, obj):
        try:
            import cupy
            if isinstance(obj, cupy.cuda.stream.Stream):
                self._cuda_stream = <cudaStream_t>(obj.ptr)
                self._owner = obj
                return
        except ImportError:
            pass
        raise TypeError(f"Cannot create stream from {type(obj)}")

    cdef void _init_with_new_cuda_stream(self) except *:
        cdef CudaStream stream = CudaStream()
        self._cuda_stream = stream.value()
        self._owner = stream

    cdef void _init_from_stream(self, Stream stream) except *:
        self._cuda_stream, self._owner = stream._cuda_stream, stream._owner


DEFAULT_STREAM = Stream._from_cudaStream_t(cuda_stream_default.value())
LEGACY_DEFAULT_STREAM = Stream._from_cudaStream_t(cuda_stream_legacy.value())
PER_THREAD_DEFAULT_STREAM = Stream._from_cudaStream_t(
    cuda_stream_per_thread.value()
)
