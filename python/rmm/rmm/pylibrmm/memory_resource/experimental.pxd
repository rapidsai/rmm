# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class CudaAsyncManagedMemoryResource(DeviceMemoryResource):
    pass
