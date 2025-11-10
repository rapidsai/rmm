# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t

from rmm.librmm.cuda_stream cimport cuda_stream_flags
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.pylibrmm.stream cimport Stream

from typing import Optional


@cython.final
cdef class CudaStreamPool:
    """
    A pool of CUDA streams for efficient stream management.

    Provides thread-safe access to a collection of CUDA stream objects.
    Successive calls may return views of identical streams.
    """

    def __cinit__(self, size_t pool_size = 16,
                  cuda_stream_flags flags = cuda_stream_flags.sync_default):
        with nogil:
            self.c_obj.reset(new cuda_stream_pool(pool_size, flags))

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    def get_stream(self, stream_id: Optional[int] = None) -> Stream:
        """
        Get a Stream from the pool (optionally by ID).

        Parameters
        ----------
        stream_id : Optional[int], optional
            The ID of the stream to get. If None, the next stream from the pool is
            returned.

        Returns
        -------
        Stream
            A non-owning Stream object from the pool
        """
        cdef size_t c_stream_id
        if stream_id is None:
            return Stream._from_cudaStream_t(
                deref(self.c_obj).get_stream().value(), owner=self)
        else:
            c_stream_id = <size_t>stream_id
            return Stream._from_cudaStream_t(
                deref(self.c_obj).get_stream(c_stream_id).value(), owner=self)

    def get_pool_size(self) -> int:
        """
        Get the pool size.

        Returns
        -------
        int
            The number of streams in the pool
        """
        return deref(self.c_obj).get_pool_size()
