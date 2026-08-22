# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from rmm.librmm.memory_resource cimport (
    cuda_async_managed_memory_resource,
    cuda_async_pinned_memory_resource,
)
# import from the private _memory_resource to avoid a circular import
from rmm.pylibrmm.memory_resource._memory_resource cimport DeviceMemoryResource


cdef class CudaAsyncManagedMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[cuda_async_managed_memory_resource] c_obj

cdef class CudaAsyncPinnedMemoryResource(DeviceMemoryResource):
    cdef unique_ptr[cuda_async_pinned_memory_resource] c_obj
