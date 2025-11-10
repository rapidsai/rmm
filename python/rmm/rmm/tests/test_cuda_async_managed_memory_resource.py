# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaAsyncManagedMemoryResource."""

import numpy as np
import pytest
from test_helpers import (
    _ASYNC_MANAGED_MEMORY_SUPPORTED,
    _allocs,
    _dtypes,
    _nelems,
    array_tester,
)

import rmm
from rmm.pylibrmm.stream import Stream


@pytest.mark.skipif(
    not _ASYNC_MANAGED_MEMORY_SUPPORTED,
    reason="CudaAsyncManagedMemoryResource requires CUDA 13.0+",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_managed_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.experimental.CudaAsyncManagedMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.skipif(
    not _ASYNC_MANAGED_MEMORY_SUPPORTED,
    reason="CudaAsyncManagedMemoryResource requires CUDA 13.0+",
)
@pytest.mark.parametrize("nelems", _nelems)
def test_cuda_async_managed_memory_resource_stream(nelems):
    mr = rmm.mr.experimental.CudaAsyncManagedMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    stream = Stream()
    expected = np.full(nelems, 5, dtype="u1")
    dbuf = rmm.DeviceBuffer.to_device(expected, stream=stream)
    result = np.asarray(dbuf.copy_to_host())
    np.testing.assert_equal(expected, result)


@pytest.mark.skipif(
    not _ASYNC_MANAGED_MEMORY_SUPPORTED,
    reason="CudaAsyncManagedMemoryResource requires CUDA 13.0+",
)
def test_cuda_async_managed_memory_resource_pool_handle():
    mr = rmm.mr.experimental.CudaAsyncManagedMemoryResource()
    pool_handle = mr.pool_handle()
    assert isinstance(pool_handle, int)
    assert pool_handle != 0
