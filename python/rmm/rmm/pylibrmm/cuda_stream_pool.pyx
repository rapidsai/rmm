# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from enum import IntEnum
from libc.stddef cimport size_t
from cython.operator cimport dereference as deref

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool, cuda_stream_flags

from rmm.pylibrmm.stream cimport Stream


class CudaStreamFlags(IntEnum):
    """
    Enumeration of CUDA stream creation flags.

    Attributes
    ----------
    SYNC_DEFAULT : int
        Created stream synchronizes with the default stream.
    NON_BLOCKING : int
        Created stream does not synchronize with the default stream.
    """
    SYNC_DEFAULT = <int>cuda_stream_flags.sync_default
    NON_BLOCKING = <int>cuda_stream_flags.non_blocking


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
            if pool_size == 0:
                raise ValueError("Pool size must be greater than zero")

            self.c_obj.reset(new cuda_stream_pool(pool_size, flags))

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    def get_stream(self) -> Stream:
        return Stream._from_cudaStream_t(
            deref(self.c_obj).get_stream().value(), owner=self)

    def get_stream_by_id(self, size_t stream_id) -> Stream:
        return Stream._from_cudaStream_t(
            deref(self.c_obj).get_stream(stream_id).value(), owner=self)

    def get_pool_size(self) -> int:
        return deref(self.c_obj).get_pool_size()
