# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Experimental memory resource features."""

from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_unique

from rmm.librmm.memory_resource cimport (
    any_device_resource,
    cuda_async_managed_memory_resource,
)
# import from the private _memory_resource to avoid a circular import
from rmm.pylibrmm.memory_resource._memory_resource cimport (
    DeviceMemoryResource,
    to_device_async_resource_ref_checked,
)


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
        # Create the typed resource and store it
        self._typed_mr = make_unique[cuda_async_managed_memory_resource]()

        # Copy into any_resource by constructing from resource_ref
        self.c_obj = any_device_resource(
            to_device_async_resource_ref_checked(self._typed_mr.get())
        )

    def pool_handle(self):
        """
        Returns the underlying CUDA memory pool handle.

        Returns
        -------
        int
            Handle to the underlying CUDA memory pool
        """
        return <uintptr_t>(self._typed_mr.get().pool_handle())
