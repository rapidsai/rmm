# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities and constants for RMM tests."""

import numpy as np
from cuda.bindings import runtime
from numba import cuda

import rmm
from rmm.allocators.numba import RMMNumbaManager

cuda.set_memory_manager(RMMNumbaManager)

# Device capability checks
_SYSTEM_MEMORY_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess,
    rmm._cuda.gpu.getDevice(),
)

_IS_INTEGRATED_MEMORY_SYSTEM = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrIntegrated, rmm._cuda.gpu.getDevice()
)

_CONCURRENT_MANAGED_ACCESS_SUPPORTED = rmm._cuda.gpu.getDeviceAttribute(
    runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess,
    rmm._cuda.gpu.getDevice(),
)

_ASYNC_MANAGED_MEMORY_SUPPORTED = (
    _CONCURRENT_MANAGED_ACCESS_SUPPORTED
    and rmm._cuda.gpu.runtimeGetVersion() >= 13000
)


def array_tester(dtype, nelem, alloc):
    """Test helper for array allocation and copy operations."""
    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = alloc.to_device(h_in)
    d_result = alloc.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    np.testing.assert_array_equal(h_result, h_in)


def assert_prefetched(buffer, device_id):
    """Check if a buffer has been prefetched to a specific device."""
    err, dev = runtime.cudaMemRangeGetAttribute(
        4,
        runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation,
        buffer.ptr,
        buffer.size,
    )
    assert err == runtime.cudaError_t.cudaSuccess
    assert dev == device_id


# Test parameter sets
_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.bool_,
]
_nelems = [1, 2, 7, 8, 9, 32, 128]
_allocs = [cuda]
