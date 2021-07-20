# Copyright (c) 2020, NVIDIA CORPORATION.
import gc
import os
import sys
from itertools import product

import numpy as np
import pytest
from numba import cuda

import rmm
import rmm._cuda.stream

if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle
else:
    import pickle

cuda.set_memory_manager(rmm.RMMNumbaManager)

_driver_version = rmm._cuda.gpu.driverGetVersion()
_runtime_version = rmm._cuda.gpu.runtimeGetVersion()
_CUDAMALLOC_ASYNC_SUPPORTED = (_driver_version >= 11020) and (
    _runtime_version >= 11020
)


@pytest.fixture(scope="function", autouse=True)
def rmm_auto_reinitialize():

    # Run the test
    yield

    # Automatically reinitialize the current memory resource after running each
    # test
    rmm.reinitialize()


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
_allocs = [cuda]


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
    assert rmm.is_initialized()
    array_tester(dtype, nelem, alloc)

    rmm.reinitialize(pool_allocator=pool, managed_memory=managed)

    assert rmm.is_initialized()

    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_csv_log(dtype, nelem, alloc, tmpdir):
    suffix = ".csv"

    base_name = str(tmpdir.join("rmm_log.csv"))
    rmm.reinitialize(logging=True, log_file_name=base_name)
    array_tester(dtype, nelem, alloc)
    rmm.mr._flush_logs()

    # Need to open separately because the device ID is appended to filename
    fname = base_name[: -len(suffix)] + ".dev0" + suffix
    try:
        with open(fname, "rb") as f:
            csv = f.read()
            assert csv.find(b"Time,Action,Pointer,Size,Stream") >= 0
    finally:
        os.remove(fname)


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


@pytest.mark.parametrize("stream", [cuda.default_stream(), cuda.stream()])
def test_rmm_pool_numba_stream(stream):
    rmm.reinitialize(pool_allocator=True)

    stream = rmm._cuda.stream.Stream(stream)
    a = rmm._lib.device_buffer.DeviceBuffer(size=3, stream=stream)

    assert a.size == 3
    assert a.ptr != 0


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


@pytest.mark.parametrize("stream", ["null", "async"])
def test_rmm_pool_cupy_allocator_with_stream(stream):
    cupy = pytest.importorskip("cupy")

    rmm.reinitialize(pool_allocator=True)
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    if stream == "null":
        stream = cupy.cuda.stream.Stream.null
    else:
        stream = cupy.cuda.stream.Stream()

    with stream:
        m = rmm.rmm_cupy_allocator(42)
        assert m.mem.size == 42
        assert m.mem.ptr != 0
        assert isinstance(m.mem._owner, rmm.DeviceBuffer)

        m = rmm.rmm_cupy_allocator(0)
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
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    stream = cupy.cuda.stream.Stream()

    stream.use()
    x = cupy.arange(10)
    del stream

    del x


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_pool_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=1 << 22,
        maximum_pool_size=1 << 23,
    )
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "upstream",
    [
        lambda: rmm.mr.CudaMemoryResource(),
        lambda: rmm.mr.ManagedMemoryResource(),
    ],
)
def test_fixed_size_memory_resource(dtype, nelem, alloc, upstream):
    mr = rmm.mr.FixedSizeMemoryResource(
        upstream(), block_size=1 << 20, blocks_to_preallocate=128
    )
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
@pytest.mark.parametrize(
    "upstream_mr",
    [
        lambda: rmm.mr.CudaMemoryResource(),
        lambda: rmm.mr.ManagedMemoryResource(),
        lambda: rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), 1 << 20
        ),
    ],
)
def test_binning_memory_resource(dtype, nelem, alloc, upstream_mr):
    upstream = upstream_mr()

    # Add fixed-size bins 256KiB, 512KiB, 1MiB, 2MiB, 4MiB
    mr = rmm.mr.BinningMemoryResource(upstream, 18, 22)

    # Test adding some explicit bin MRs
    fixed_mr = rmm.mr.FixedSizeMemoryResource(upstream, 1 << 10)
    cuda_mr = rmm.mr.CudaMemoryResource()
    mr.add_bin(1 << 10, fixed_mr)  # 1KiB bin
    mr.add_bin(1 << 23, cuda_mr)  # 8MiB bin

    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


def test_reinitialize_max_pool_size():
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size=1 << 23
    )
    rmm.DeviceBuffer().resize((1 << 23) - 1)


def test_reinitialize_max_pool_size_exceeded():
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size=1 << 23
    )
    with pytest.raises(MemoryError):
        rmm.DeviceBuffer().resize(1 << 24)


def test_reinitialize_initial_pool_size_gt_max():
    with pytest.raises(RuntimeError) as e:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=1 << 11,
            maximum_pool_size=1 << 10,
        )
    assert "Initial pool size exceeds the maximum pool size" in str(e.value)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_enable_disable_logging(dtype, nelem, alloc, tmpdir):
    suffix = ".csv"

    base_name = str(tmpdir.join("rmm_log.csv"))

    rmm.enable_logging(log_file_name=base_name)
    print(rmm.mr.get_per_device_resource(0))
    array_tester(dtype, nelem, alloc)
    rmm.mr._flush_logs()

    # Need to open separately because the device ID is appended to filename
    fname = base_name[: -len(suffix)] + ".dev0" + suffix
    try:
        with open(fname, "rb") as f:
            csv = f.read()
            assert csv.find(b"Time,Action,Pointer,Size,Stream") >= 0
    finally:
        os.remove(fname)

    rmm.disable_logging()


def test_mr_devicebuffer_lifetime():
    # Test ensures MR/Stream lifetime is longer than DeviceBuffer. Even if all
    # references go out of scope
    # Create new Pool MR
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
    )

    # Creates a new non-default stream
    stream = rmm._cuda.stream.Stream()

    # Allocate DeviceBuffer with Pool and Stream
    a = rmm.DeviceBuffer(size=10, stream=stream)

    # Change current MR. Will cause Pool to go out of scope
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    # Force collection to ensure objects are cleaned up
    gc.collect()

    # Delete a. Used to crash before. Pool MR should still be alive
    del a


def test_mr_upstream_lifetime():
    # Simple test to ensure upstream MRs are deallocated before downstream MR
    cuda_mr = rmm.mr.CudaMemoryResource()

    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr)

    # Delete cuda_mr first. Should be kept alive by pool_mr
    del cuda_mr
    del pool_mr


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED, reason="cudaMallocAsync not supported",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED, reason="cudaMallocAsync not supported",
)
@pytest.mark.parametrize("nelems", _nelems)
def test_cuda_async_memory_resource_stream(nelems):
    # test that using CudaAsyncMemoryResource
    # with a non-default stream works
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    stream = rmm._cuda.stream.Stream()
    expected = np.full(nelems, 5, dtype="u1")
    dbuf = rmm.DeviceBuffer.to_device(expected, stream=stream)
    result = np.asarray(dbuf.copy_to_host())
    np.testing.assert_equal(expected, result)


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED, reason="cudaMallocAsync not supported",
)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_cuda_async_memory_resource_threshold(nelem, alloc):
    # initial pool size == 0
    mr = rmm.mr.CudaAsyncMemoryResource(
        initial_pool_size=0, release_threshold=nelem
    )
    rmm.mr.set_current_device_resource(mr)
    array_tester("u1", nelem, alloc)  # should not trigger release
    array_tester("u1", 2 * nelem, alloc)  # should trigger release


def test_statistics_resource_adaptor():

    cuda_mr = rmm.mr.CudaMemoryResource()

    mr = rmm.mr.StatisticsResourceAdaptor(cuda_mr)

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert mr.allocation_counts == {
        "current_bytes": 5000,
        "current_count": 5,
        "peak_bytes": 10000,
        "peak_count": 10,
        "total_bytes": 10000,
        "total_count": 10,
    }

    # Push a new Tracking adaptor
    mr2 = rmm.mr.StatisticsResourceAdaptor(mr)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.allocation_counts == {
        "current_bytes": 2000,
        "current_count": 2,
        "peak_bytes": 2000,
        "peak_count": 2,
        "total_bytes": 2000,
        "total_count": 2,
    }
    assert mr.allocation_counts == {
        "current_bytes": 7000,
        "current_count": 7,
        "peak_bytes": 10000,
        "peak_count": 10,
        "total_bytes": 12000,
        "total_count": 12,
    }

    del buffers
    gc.collect()

    assert mr2.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 2000,
        "peak_count": 2,
        "total_bytes": 2000,
        "total_count": 2,
    }
    assert mr.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 10000,
        "peak_count": 10,
        "total_bytes": 12000,
        "total_count": 12,
    }


def test_tracking_resource_adaptor():

    cuda_mr = rmm.mr.CudaMemoryResource()

    mr = rmm.mr.TrackingResourceAdaptor(cuda_mr, capture_stacks=True)

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert mr.get_allocated_bytes() == 5000

    # Push a new Tracking adaptor
    mr2 = rmm.mr.TrackingResourceAdaptor(mr, capture_stacks=True)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.get_allocated_bytes() == 2000
    assert mr.get_allocated_bytes() == 7000

    # Ensure we get back a non-empty string for the allocations
    assert len(mr.get_outstanding_allocations_str()) > 0

    del buffers
    gc.collect()

    assert mr2.get_allocated_bytes() == 0
    assert mr.get_allocated_bytes() == 0

    # make sure the allocations string is now empty
    assert len(mr2.get_outstanding_allocations_str()) == 0
    assert len(mr.get_outstanding_allocations_str()) == 0
