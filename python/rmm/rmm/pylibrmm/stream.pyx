# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uintptr_t
from libcpp cimport bool

from rmm.librmm.cuda_stream_view cimport (
    cuda_stream_default,
    cuda_stream_legacy,
    cuda_stream_per_thread,
    cuda_stream_view,
)
from rmm.pylibrmm.cuda_stream cimport CudaStream


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
        else:
            if hasattr(obj, "__cuda_stream__"):
                self._init_from_cuda_stream_protocol(obj)
            else:
                try:
                    self._init_from_numba_stream(obj)
                except TypeError:
                    self._init_from_cupy_stream(obj)

    @staticmethod
    cdef Stream _from_cudaStream_t(cudaStream_t s, object owner=None) except *:
        """
        Construct a Stream from a cudaStream_t.
        """
        cdef Stream obj = Stream.__new__(Stream)
        obj._cuda_stream = s
        obj._owner = owner
        return obj

    def __cuda_stream__(self):
        # Implementation of the CUDA stream protocol
        # https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
        return (0, int(<uintptr_t>self._cuda_stream))

    cdef cuda_stream_view view(self) noexcept nogil:
        """
        Generate a rmm::cuda_stream_view from this Stream instance
        """
        return cuda_stream_view(<cudaStream_t><uintptr_t>(self._cuda_stream))

    cdef void c_synchronize(self) except * nogil:
        """
        Synchronize the CUDA stream.
        This function *must* be called in a `with nogil` block
        """
        with nogil:
            self.view().synchronize()

    def synchronize(self):
        """
        Synchronize the CUDA stream
        """
        with nogil:
            self.c_synchronize()

    cdef bool c_is_default(self) noexcept nogil:
        """
        Check if we are the default CUDA stream
        """
        return self.view().is_default()

    def is_default(self):
        """
        Check if we are the default CUDA stream
        """
        return self.c_is_default()

    def _init_from_cuda_stream_protocol(self, obj):
        """
        Initialize `self` from an object implementing the CUDA Stream Protocol.
        """
        version, ptr = obj.__cuda_stream__()
        if version == 0:
            self._cuda_stream = <cudaStream_t><uintptr_t>(ptr)
            self._owner = obj
        else:
            raise NotImplementedError(
                f"RMM does not currently support the CUDA Stream Protocol "
                f"version: '{version}'."
            )

    def _init_from_numba_stream(self, obj):
        try:
            from numba import cuda
            if isinstance(obj, cuda.cudadrv.driver.Stream):
                self._cuda_stream = <cudaStream_t><uintptr_t>(int(obj))
                self._owner = obj
                return
        except ImportError:
            pass
        raise TypeError(f"Cannot create stream from {type(obj)}")

    def _init_from_cupy_stream(self, obj):
        try:
            import cupy
            if isinstance(obj, (cupy.cuda.stream.Stream,
                                cupy.cuda.stream.ExternalStream)):
                self._cuda_stream = <cudaStream_t><uintptr_t>(obj.ptr)
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
