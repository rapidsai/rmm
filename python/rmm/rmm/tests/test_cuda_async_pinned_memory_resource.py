# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaAsyncPinnedMemoryResource."""

import ctypes

import numpy as np
from numba import cuda

import rmm
from rmm.pylibrmm.stream import CudaStreamFlags, Stream


@cuda.jit
def _increment(values):
    i = cuda.grid(1)
    if i < values.size:
        values[i] += 1


def test_cuda_async_pinned_memory_resource_pool_handle():
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    other = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()

    pool_handle = mr.pool_handle()
    assert isinstance(pool_handle, int)
    assert pool_handle != 0
    assert other.pool_handle() == pool_handle


def test_cuda_async_pinned_memory_resource_non_default_stream():
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    stream = Stream(flags=CudaStreamFlags.NON_BLOCKING)

    ptr = mr.allocate(1024, stream=stream)
    assert ptr != 0
    mr.deallocate(ptr, 1024, stream=stream)
    stream.synchronize()


def test_cuda_async_pinned_memory_resource_host_and_device_access():
    mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    stream = Stream(flags=CudaStreamFlags.NON_BLOCKING)
    size = 128
    buffer = rmm.DeviceBuffer(size=size, stream=stream, mr=mr)

    expected = np.arange(size, dtype="u1")
    buffer.copy_from_host(expected, stream=stream)
    stream.synchronize()

    host_view = np.ctypeslib.as_array(
        (ctypes.c_uint8 * size).from_address(buffer.ptr)
    )
    np.testing.assert_array_equal(host_view, expected)

    # Numba's EMM copy paths require true device allocations. Construct a
    # device view of this existing allocation to test kernel access without
    # asking Numba to allocate from the pinned memory resource.
    device_view = cuda.as_cuda_array(buffer)
    numba_stream = cuda.external_stream(stream.__cuda_stream__()[1])
    _increment[1, 128, numba_stream](device_view)
    stream.synchronize()

    expected += 1
    np.testing.assert_array_equal(host_view, expected)


def test_cuda_async_pinned_memory_resource_device_buffer_copies():
    pinned_mr = rmm.mr.experimental.CudaAsyncPinnedMemoryResource()
    device_mr = rmm.mr.CudaMemoryResource()
    stream = Stream(flags=CudaStreamFlags.NON_BLOCKING)
    size = 251

    pinned_source = rmm.DeviceBuffer(size=size, stream=stream, mr=pinned_mr)
    device_buffer = rmm.DeviceBuffer(size=size, stream=stream, mr=device_mr)
    pinned_destination = rmm.DeviceBuffer(
        size=size, stream=stream, mr=pinned_mr
    )
    pinned_copy = rmm.DeviceBuffer(size=size, stream=stream, mr=pinned_mr)

    expected = np.arange(size, dtype="u1")
    pinned_source.copy_from_host(expected, stream=stream)
    device_buffer.copy_from_device(pinned_source, stream=stream)
    pinned_destination.copy_from_device(device_buffer, stream=stream)
    pinned_copy.copy_from_device(pinned_destination, stream=stream)
    result = pinned_copy.copy_to_host(stream=stream)
    stream.synchronize()

    np.testing.assert_array_equal(result, expected)
