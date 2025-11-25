# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaAsyncPinnedMemoryResource."""

import numpy as np
import pytest
from test_helpers import (
    _ASYNC_PINNED_MEMORY_SUPPORTED,
    _allocs,
    _dtypes,
    _nelems,
    array_tester,
)

import rmm
from rmm.pylibrmm.stream import Stream


@pytest.mark.skipif(
    not _ASYNC_PINNED_MEMORY_SUPPORTED,
    reason="CudaAsyncPinnedMemoryResource requires CUDA 12.6+",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_pinned_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.skipif(
    not _ASYNC_PINNED_MEMORY_SUPPORTED,
    reason="CudaAsyncPinnedMemoryResource requires CUDA 12.6+",
)
@pytest.mark.parametrize("nelems", _nelems)
def test_cuda_async_pinned_memory_resource_stream(nelems):
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    stream = Stream()
    expected = np.full(nelems, 5, dtype="u1")
    dbuf = rmm.DeviceBuffer.to_device(expected, stream=stream)
    result = np.asarray(dbuf.copy_to_host())
    np.testing.assert_equal(expected, result)


@pytest.mark.skipif(
    not _ASYNC_PINNED_MEMORY_SUPPORTED,
    reason="CudaAsyncPinnedMemoryResource requires CUDA 12.6+",
)
def test_cuda_async_pinned_memory_resource_pool_handle():
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    pool_handle = mr.pool_handle()
    assert isinstance(pool_handle, int)
    assert pool_handle != 0


@pytest.mark.skipif(
    not _ASYNC_PINNED_MEMORY_SUPPORTED,
    reason="CudaAsyncPinnedMemoryResource requires CUDA 12.6+",
)
def test_cuda_async_pinned_memory_resource_host_access():
    """Test that pinned memory allocated by the resource is accessible from host."""
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Allocate a buffer
    expected = np.full(100, 42, dtype="u1")
    dbuf = rmm.DeviceBuffer.to_device(expected)

    # Verify host can access the data
    result = np.asarray(dbuf.copy_to_host())
    np.testing.assert_equal(expected, result)
