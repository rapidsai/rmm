# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CuPy integration with RMM."""

import pytest

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator


def test_rmm_cupy_allocator():
    cupy = pytest.importorskip("cupy")

    m = rmm_cupy_allocator(42)
    assert m.mem.size == 42
    assert m.mem.ptr != 0
    assert isinstance(m.mem._owner, rmm.DeviceBuffer)

    m = rmm_cupy_allocator(0)
    assert m.mem.size == 0
    assert m.mem.ptr == 0
    assert isinstance(m.mem._owner, rmm.DeviceBuffer)

    cupy.cuda.set_allocator(rmm_cupy_allocator)
    a = cupy.arange(10)
    assert isinstance(a.data.mem._owner, rmm.DeviceBuffer)


@pytest.mark.parametrize("stream", ["null", "async"])
def test_rmm_pool_cupy_allocator_with_stream(stream):
    cupy = pytest.importorskip("cupy")

    rmm.reinitialize(pool_allocator=True)
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    if stream == "null":
        stream = cupy.cuda.stream.Stream.null
    else:
        stream = cupy.cuda.stream.Stream()

    with stream:
        m = rmm_cupy_allocator(42)
        assert m.mem.size == 42
        assert m.mem.ptr != 0
        assert isinstance(m.mem._owner, rmm.DeviceBuffer)

        m = rmm_cupy_allocator(0)
        assert m.mem.size == 0
        assert m.mem.ptr == 0
        assert isinstance(m.mem._owner, rmm.DeviceBuffer)

        a = cupy.arange(10)
        assert isinstance(a.data.mem._owner, rmm.DeviceBuffer)

    # Deleting all allocations known by the RMM pool is required
    # before rmm.reinitialize(), otherwise it may segfault.
    del a

    rmm.reinitialize()


def test_rmm_pool_cupy_allocator_stream_lifetime():
    cupy = pytest.importorskip("cupy")

    rmm.reinitialize(pool_allocator=True)
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    stream = cupy.cuda.stream.Stream()

    stream.use()
    x = cupy.arange(10)
    del stream

    del x
