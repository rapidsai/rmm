import pickle
import sys
from itertools import product

import numpy as np
import pytest
from numba import cuda

import rmm

cuda.set_memory_manager(rmm.RMMNumbaManager)


def array_tester(dtype, nelem, alloc):
    # data
    h_in = np.full(nelem, 3.2, dtype)
    h_result = np.empty(nelem, dtype)

    d_in = alloc.to_device(h_in)
    d_result = alloc.device_array_like(d_in)

    d_result.copy_to_device(d_in)
    h_result = d_result.copy_to_host()

    np.testing.assert_array_equal(h_result, h_in)


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
_allocs = [cuda, rmm]


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_alloc(dtype, nelem, alloc):
    array_tester(dtype, nelem, alloc)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "managed, pool", list(product([False, True], [False, True]))
)
def test_rmm_modes(dtype, nelem, alloc, managed, pool):
    rmm.reinitialize(pool_allocator=pool, managed_memory=managed)

    assert rmm.is_initialized()

    array_tester(dtype, nelem, alloc)


def test_uninitialized():
    rmm._finalize()
    assert not rmm.is_initialized()
    rmm.reinitialize()  # so further tests will pass


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
def test_rmm_csv_log(dtype, nelem):
    # data
    h_in = np.full(nelem, 3.2, dtype)

    d_in = cuda.to_device(h_in)
    d_result = cuda.device_array_like(d_in)

    d_result.copy_to_device(d_in)

    csv = rmm.csv_log()

    assert (
        csv.find(
            "Event Type,Device ID,Address,Stream,Size (bytes),"
            "Free Memory,Total Memory,Current Allocs,Start,End,"
            "Elapsed,Location"
        )
        >= 0
    )


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
    assert sys.getsizeof(b) == b.size

    # Test `__cuda_array_interface__`
    keyset = {"data", "shape", "strides", "typestr", "version"}
    assert isinstance(b.__cuda_array_interface__, dict)
    assert set(b.__cuda_array_interface__) == keyset
    assert b.__cuda_array_interface__["data"] == (b.ptr, False)
    assert b.__cuda_array_interface__["shape"] == (b.size,)
    assert b.__cuda_array_interface__["strides"] == (1,)
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
        lambda: rmm.to_device(np.array([97, 98, 99], dtype="u1")),
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
    if pickle.HIGHEST_PROTOCOL >= 5:
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


def test_rmm_cupy_allocator():
    cupy = pytest.importorskip("cupy")

    m = rmm.rmm_cupy_allocator(42)
    assert m.mem.size == 42
    assert m.mem.ptr != 0
    assert isinstance(m.mem._owner, rmm.DeviceBuffer)

    m = rmm.rmm_cupy_allocator(0)
    assert m.mem.size == 0
    assert m.mem.ptr == 0
    assert isinstance(m.mem._owner, rmm.DeviceBuffer)

    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
    a = cupy.arange(10)
    assert isinstance(a.data.mem._owner, rmm.DeviceBuffer)


def test_rmm_getinfo():
    meminfo = rmm.get_info()
    # Basic sanity checks of returned values
    assert meminfo.free >= 0
    assert meminfo.total >= 0
    assert meminfo.free <= meminfo.total


def test_rmm_getinfo_uninitialized():
    rmm._finalize()

    with pytest.raises(rmm.RMMError):
        rmm.get_info()

    rmm.reinitialize()
