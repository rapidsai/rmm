# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cuda.bindings.cyruntime cimport cudaStream_t
from enum import IntEnum
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool, cuda_stream_flags
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

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
    
    def __cinit__(self, size_t pool_size = 16, cuda_stream_flags flags = cuda_stream_flags.sync_default):
        """
        Construct a new CUDA stream pool.
        
        Parameters
        ----------
        pool_size : size_t, optional
            The number of streams in the pool. Default is 16.
        flags : CudaStreamFlags, optional  
            Flags used when creating streams in the pool. Must be 
            CudaStreamFlags.SYNC_DEFAULT or CudaStreamFlags.NON_BLOCKING.
            Default is CudaStreamFlags.SYNC_DEFAULT.
        
        Raises
        ------
        ValueError
            If pool_size is zero.
        """
        cdef cuda_stream_flags c_flags = flags

        if pool_size == 0:
            raise ValueError("Pool size must be greater than zero")
            
        with nogil:
            self.c_obj.reset(new cuda_stream_pool(pool_size, c_flags))

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    def get_stream(self, stream_id: Optional[int] = None) -> Stream:
        """
        Get a CUDA stream from the pool.
        
        This method is thread-safe with respect to other calls to the same method.
        
        Parameters
        ----------
        stream_id : size_t, optional
            Unique identifier for the desired stream. If None, returns
            the next available stream from the pool.
            
        Returns
        -------
        Stream
            The Stream object representing the CUDA stream.

        Raises
        ------
        ValueError
            If stream_id is out of range.
        """          
        cdef cuda_stream_view stream_view 
        
        if stream_id is None:
            stream_view = deref(self.c_obj).get_stream()
        else:
            if stream_id >= deref(self.c_obj).get_pool_size():
                raise ValueError(f"Stream id {stream_id} is out of range")
            stream_view = deref(self.c_obj).get_stream(stream_id)
                
        return Stream._from_cudaStream_t(stream_view.value(), owner=self)

    def get_pool_size(self) -> int:
        """
        Get the number of streams in the pool.
        
        This method is thread-safe.
        
        Returns
        -------
        int
            The number of streams in the pool.
        """
        return deref(self.c_obj).get_pool_size()
