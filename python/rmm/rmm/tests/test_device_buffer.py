# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeviceBuffer class."""

import copy
import pickle
from itertools import product

import numpy as np
import pytest
from cuda.bindings import runtime
from numba import cuda
from test_helpers import (
    _CONCURRENT_MANAGED_ACCESS_SUPPORTED,
    assert_prefetched,
)

import rmm


@pytest.mark.parametrize("size", [0, 5])
def test_rmm_device_buffer(size):
    b = rmm.DeviceBuffer(size=size)

    # Test some properties
    if size:
        assert b.ptr != 0
        assert b.size == size
    else:
        assert b.ptr == 0
        assert b.size == 0
    assert len(b) == b.size
    assert b.nbytes == b.size
    assert b.capacity() >= b.size
    assert b.__sizeof__() == b.size

    # Test `__cuda_array_interface__`
    keyset = {"data", "shape", "strides", "typestr", "version"}
    assert isinstance(b.__cuda_array_interface__, dict)
    assert set(b.__cuda_array_interface__) == keyset
    assert b.__cuda_array_interface__["data"] == (b.ptr, False)
    assert b.__cuda_array_interface__["shape"] == (b.size,)
    assert b.__cuda_array_interface__["strides"] is None
    assert b.__cuda_array_interface__["typestr"] == "|u1"
    assert b.__cuda_array_interface__["version"] == 0

    # Test conversion to bytes
    s = b.tobytes()
    assert isinstance(s, bytes)
    assert len(s) == len(b)

    # Test conversion from bytes
    b2 = rmm.DeviceBuffer.to_device(s)
    assert isinstance(b2, rmm.DeviceBuffer)
    assert len(b2) == len(s)

    # Test resizing
    b.resize(2)
    assert b.size == 2
    assert b.capacity() >= b.size


@pytest.mark.parametrize(
    "hb",
    [
        b"abc",
        bytearray(b"abc"),
        memoryview(b"abc"),
        np.asarray(memoryview(b"abc")),
        np.arange(3, dtype="u1"),
    ],
)
def test_rmm_device_buffer_memoryview_roundtrip(hb):
    mv = memoryview(hb)
    db = rmm.DeviceBuffer.to_device(hb)
    hb2 = db.copy_to_host()
    assert isinstance(hb2, np.ndarray)
    mv2 = memoryview(hb2)
    assert mv == mv2
    hb3a = bytearray(mv.nbytes)
    hb3b = db.copy_to_host(hb3a)
    assert hb3a is hb3b
    mv3 = memoryview(hb3b)
    assert mv == mv3
    hb4a = np.empty_like(mv)
    hb4b = db.copy_to_host(hb4a)
    assert hb4a is hb4b
    mv4 = memoryview(hb4b)
    assert mv == mv4


@pytest.mark.parametrize(
    "hb",
    [
        None,
        "abc",
        123,
        b"",
        np.ones((2,), "u2"),
        np.ones((2, 2), "u1"),
        np.ones(4, "u1")[::2],
        b"abc",
        bytearray(b"abc"),
        memoryview(b"abc"),
        np.asarray(memoryview(b"abc")),
        np.arange(3, dtype="u1"),
    ],
)
def test_rmm_device_buffer_bytes_roundtrip(hb):
    try:
        mv = memoryview(hb)
    except TypeError:
        with pytest.raises(TypeError):
            rmm.DeviceBuffer.to_device(hb)
    else:
        if mv.format != "B":
            with pytest.raises(ValueError):
                rmm.DeviceBuffer.to_device(hb)
        elif len(mv.strides) != 1:
            with pytest.raises(ValueError):
                rmm.DeviceBuffer.to_device(hb)
        elif mv.strides[0] != 1:
            with pytest.raises(ValueError):
                rmm.DeviceBuffer.to_device(hb)
        else:
            db = rmm.DeviceBuffer.to_device(hb)
            hb2 = db.tobytes()
            mv2 = memoryview(hb2)
            assert mv == mv2
            hb3 = bytes(db)
            mv3 = memoryview(hb3)
            assert mv == mv3


@pytest.mark.parametrize(
    "hb",
    [
        b"abc",
        bytearray(b"abc"),
        memoryview(b"abc"),
        np.asarray(memoryview(b"abc")),
        np.array([97, 98, 99], dtype="u1"),
    ],
)
def test_rmm_device_buffer_copy_from_host(hb):
    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    db.copy_from_host(hb)

    expected = np.array([97, 98, 99, 0, 0, 0, 0, 0, 0, 0], dtype="u1")
    result = db.copy_to_host()

    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    "cuda_ary",
    [
        lambda: rmm.DeviceBuffer.to_device(b"abc"),
        lambda: cuda.to_device(np.array([97, 98, 99], dtype="u1")),
    ],
)
def test_rmm_device_buffer_copy_from_device(cuda_ary):
    cuda_ary = cuda_ary()
    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    db.copy_from_device(cuda_ary)

    expected = np.array([97, 98, 99, 0, 0, 0, 0, 0, 0, 0], dtype="u1")
    result = db.copy_to_host()

    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize("hb", [b"", b"123", b"abc"])
def test_rmm_device_buffer_pickle_roundtrip(hb):
    db = rmm.DeviceBuffer.to_device(hb)
    pb = pickle.dumps(db)
    del db
    db2 = pickle.loads(pb)
    hb2 = db2.tobytes()
    assert hb == hb2
    # out-of-band
    db = rmm.DeviceBuffer.to_device(hb)
    buffers = []
    pb2 = pickle.dumps(db, protocol=5, buffer_callback=buffers.append)
    del db
    assert len(buffers) == 1
    assert isinstance(buffers[0], pickle.PickleBuffer)
    assert bytes(buffers[0]) == hb
    db3 = pickle.loads(pb2, buffers=buffers)
    hb3 = db3.tobytes()
    assert hb3 == hb


@pytest.mark.parametrize(
    "managed, pool", list(product([False, True], [False, True]))
)
def test_rmm_device_buffer_prefetch(pool, managed):
    rmm.reinitialize(pool_allocator=pool, managed_memory=managed)
    db = rmm.DeviceBuffer.to_device(np.zeros(256, dtype="u1"))
    if managed and _CONCURRENT_MANAGED_ACCESS_SUPPORTED:
        assert_prefetched(db, runtime.cudaInvalidDeviceId)
    db.prefetch()  # just test that it doesn't throw
    if managed and _CONCURRENT_MANAGED_ACCESS_SUPPORTED:
        err, device_id = runtime.cudaGetDevice()
        assert err == runtime.cudaError_t.cudaSuccess
        assert_prefetched(db, device_id)


@pytest.mark.parametrize(
    "cuda_ary",
    [
        lambda: rmm.DeviceBuffer.to_device(b"abc"),
        lambda: cuda.to_device(np.array([97, 98, 99, 0, 0], dtype="u1")),
    ],
)
@pytest.mark.parametrize(
    "make_copy", [lambda db: db.copy(), lambda db: copy.copy(db)]
)
def test_rmm_device_buffer_copy(cuda_ary, make_copy):
    cuda_ary = cuda_ary()
    db = rmm.DeviceBuffer.to_device(np.zeros(5, dtype="u1"))
    db.copy_from_device(cuda_ary)
    db_copy = make_copy(db)

    assert db is not db_copy
    assert db.ptr != db_copy.ptr
    assert len(db) == len(db_copy)

    expected = np.array([97, 98, 99, 0, 0], dtype="u1")
    result = db_copy.copy_to_host()

    np.testing.assert_equal(expected, result)


# Tests for stream=None validation (PR #2120)
def test_device_buffer_init_stream_none():
    """Test that DeviceBuffer.__init__ raises TypeError for stream=None"""
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        rmm.DeviceBuffer(size=10, stream=None)


def test_device_buffer_to_device_stream_none():
    """Test that DeviceBuffer.to_device raises TypeError for stream=None"""
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        rmm.DeviceBuffer.to_device(b"abc", stream=None)


def test_device_buffer_copy_to_host_stream_none():
    """Test that DeviceBuffer.copy_to_host raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(b"abc")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.copy_to_host(stream=None)


def test_device_buffer_copy_from_host_stream_none():
    """Test that DeviceBuffer.copy_from_host raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.copy_from_host(b"abc", stream=None)


def test_device_buffer_copy_from_device_stream_none():
    """Test that DeviceBuffer.copy_from_device raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    cuda_ary = rmm.DeviceBuffer.to_device(b"abc")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.copy_from_device(cuda_ary, stream=None)


def test_device_buffer_tobytes_stream_none():
    """Test that DeviceBuffer.tobytes raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(b"abc")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.tobytes(stream=None)


def test_device_buffer_reserve_stream_none():
    """Test that DeviceBuffer.reserve raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(b"abc")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.reserve(100, stream=None)


def test_device_buffer_resize_stream_none():
    """Test that DeviceBuffer.resize raises TypeError for stream=None"""
    db = rmm.DeviceBuffer.to_device(b"abc")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        db.resize(10, stream=None)


def test_to_device_stream_none():
    """Test that to_device function raises TypeError for stream=None"""
    # Import the module-level function
    from rmm import DeviceBuffer

    with pytest.raises(TypeError, match="stream argument cannot be None"):
        DeviceBuffer.to_device(b"abc", stream=None)


def test_copy_ptr_to_host_stream_none():
    """Test that copy_ptr_to_host raises TypeError for stream=None"""
    from rmm.pylibrmm.device_buffer import copy_ptr_to_host

    db = rmm.DeviceBuffer.to_device(b"abc")
    hb = bytearray(3)
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        copy_ptr_to_host(db.ptr, hb, stream=None)


def test_copy_host_to_ptr_stream_none():
    """Test that copy_host_to_ptr raises TypeError for stream=None"""
    from rmm.pylibrmm.device_buffer import copy_host_to_ptr

    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    hb = np.array([97, 98, 99], dtype="u1")
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        copy_host_to_ptr(hb, db.ptr, stream=None)


def test_copy_device_to_ptr_stream_none():
    """Test that copy_device_to_ptr raises TypeError for stream=None"""
    from rmm.pylibrmm.device_buffer import copy_device_to_ptr

    db_src = rmm.DeviceBuffer.to_device(b"abc")
    db_dst = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        copy_device_to_ptr(db_src.ptr, db_dst.ptr, 3, stream=None)


def test_memory_resource_allocate_stream_none():
    """Test that DeviceMemoryResource.allocate raises TypeError for stream=None"""
    mr = rmm.mr.get_current_device_resource()
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        mr.allocate(1024, stream=None)


def test_memory_resource_deallocate_stream_none():
    """Test that DeviceMemoryResource.deallocate raises TypeError for stream=None"""
    mr = rmm.mr.get_current_device_resource()
    ptr = mr.allocate(1024)
    with pytest.raises(TypeError, match="stream argument cannot be None"):
        mr.deallocate(ptr, 1024, stream=None)
