# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# import from the private _memory_resource to avoid a circular import
from rmm.pylibrmm.memory_resource._memory_resource cimport DeviceMemoryResource


cdef class CudaAsyncManagedMemoryResource(DeviceMemoryResource):
    pass
