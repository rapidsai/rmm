# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Experimental memory resource features."""

from libc.stdint cimport uintptr_t

# import from the private _memory_resource to avoid a circular import
from rmm.pylibrmm.memory_resource._memory_resource cimport DeviceMemoryResource

from rmm.librmm.memory_resource cimport cuda_async_managed_memory_resource


cdef class CudaAsyncManagedMemoryResource(DeviceMemoryResource):
    """
    Memory resource that uses ``cudaMallocFromPoolAsync``/``cudaFreeAsync`` for
    allocation/deallocation with a managed memory pool.

    This resource uses the default managed memory pool for the current device.
    Managed memory can be accessed from both the host and device.

    Requires CUDA 13.0 or higher and support for concurrent managed access
    (not supported on WSL2).
    """
    def __cinit__(self):
        self.c_obj.reset(
            new cuda_async_managed_memory_resource()
        )

    def pool_handle(self):
        """
        Returns the underlying CUDA memory pool handle.

        Returns
        -------
        int
            Handle to the underlying CUDA memory pool
        """
        cdef cuda_async_managed_memory_resource* c_mr = \
            <cuda_async_managed_memory_resource*>self.c_obj.get()
        return <uintptr_t>c_mr.pool_handle()
