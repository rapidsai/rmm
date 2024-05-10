# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import gc
import os
import pickle
import warnings
from itertools import product

import numpy as np
import pytest
from numba import cuda

import rmm
import rmm._cuda.stream
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager

cuda.set_memory_manager(RMMNumbaManager)

_driver_version = rmm._cuda.gpu.driverGetVersion()
_runtime_version = rmm._cuda.gpu.runtimeGetVersion()
_CUDAMALLOC_ASYNC_SUPPORTED = (_driver_version >= 11020) and (
    _runtime_version >= 11020
)


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
    # It is necessary to verify that it also works when using an upstream :
    # here a Pool MR with the current MR as upstream
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


def test_device_buffer_with_mr():
    allocations = []
    base = rmm.mr.CudaMemoryResource()
    rmm.mr.set_current_device_resource(base)

    def alloc_cb(size, stream, *, base):
        allocations.append(size)
        return base.allocate(size, stream)

    def dealloc_cb(ptr, size, stream, *, base):
        return base.deallocate(ptr, size, stream)

    cb_mr = rmm.mr.CallbackMemoryResource(
        functools.partial(alloc_cb, base=base),
        functools.partial(dealloc_cb, base=base),
    )
    rmm.DeviceBuffer(size=10)
    assert len(allocations) == 0
    buf = rmm.DeviceBuffer(size=256, mr=cb_mr)
    assert len(allocations) == 1
    assert allocations[0] == 256
    del cb_mr
    gc.collect()
    del buf


def test_mr_upstream_lifetime():
    # Simple test to ensure upstream MRs are deallocated before downstream MR
    cuda_mr = rmm.mr.CudaMemoryResource()

    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr)

    # Delete cuda_mr first. Should be kept alive by pool_mr
    del cuda_mr
    del pool_mr


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED,
    reason="cudaMallocAsync not supported",
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
    not _CUDAMALLOC_ASYNC_SUPPORTED,
    reason="cudaMallocAsync not supported",
)
def test_cuda_async_memory_resource_ipc():
    # TODO: We don't have a great way to check if IPC is supported in Python,
    # without using the C++ function
    # rmm::detail::async_alloc::is_export_handle_type_supported. We can't
    # accurately test driver and runtime versions for this via Python because
    # cuda-python always has the IPC handle enum defined (which normally
    # requires a CUDA 11.3 runtime) and the cuda-compat package in Docker
    # containers prevents us from assuming that the driver we see actually
    # supports IPC handles even if its reported version is new enough (we may
    # see a newer driver than what is present on the host). We can only know
    # the expected behavior by checking the C++ function mentioned above, which
    # is then a redundant check because the CudaAsyncMemoryResource constructor
    # follows the same logic. Therefore, we cannot easily ensure this test
    # passes in certain expected configurations -- we can only ensure that if
    # it fails, it fails in a predictable way.
    try:
        mr = rmm.mr.CudaAsyncMemoryResource(enable_ipc=True)
    except RuntimeError as e:
        # CUDA 11.3 is required for IPC memory handle support
        assert str(e).endswith(
            "Requested IPC memory handle type not supported"
        )
    else:
        rmm.mr.set_current_device_resource(mr)
        assert rmm.mr.get_current_device_resource_type() is type(mr)


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED,
    reason="cudaMallocAsync not supported",
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
    not _CUDAMALLOC_ASYNC_SUPPORTED,
    reason="cudaMallocAsync not supported",
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


@pytest.mark.parametrize(
    "mr",
    [
        rmm.mr.CudaMemoryResource,
        pytest.param(
            rmm.mr.CudaAsyncMemoryResource,
            marks=pytest.mark.skipif(
                not _CUDAMALLOC_ASYNC_SUPPORTED,
                reason="cudaMallocAsync not supported",
            ),
        ),
    ],
)
def test_limiting_resource_adaptor(mr):
    cuda_mr = mr()

    allocation_limit = 1 << 20
    num_buffers = 2
    buffer_size = allocation_limit // num_buffers

    mr = rmm.mr.LimitingResourceAdaptor(
        cuda_mr, allocation_limit=allocation_limit
    )
    assert mr.get_allocation_limit() == allocation_limit

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=buffer_size) for _ in range(num_buffers)]

    assert mr.get_allocated_bytes() == sum(b.size for b in buffers)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=1)


def test_statistics_resource_adaptor(stats_mr):

    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert stats_mr.allocation_counts == {
        "current_bytes": 5040,
        "current_count": 5,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 10080,
        "total_count": 10,
    }

    # Push a new Tracking adaptor
    mr2 = rmm.mr.StatisticsResourceAdaptor(stats_mr)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.allocation_counts == {
        "current_bytes": 2016,
        "current_count": 2,
        "peak_bytes": 2016,
        "peak_count": 2,
        "total_bytes": 2016,
        "total_count": 2,
    }
    assert stats_mr.allocation_counts == {
        "current_bytes": 7056,
        "current_count": 7,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 12096,
        "total_count": 12,
    }

    del buffers
    gc.collect()

    assert mr2.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 2016,
        "peak_count": 2,
        "total_bytes": 2016,
        "total_count": 2,
    }
    assert stats_mr.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 12096,
        "total_count": 12,
    }
    gc.collect()


def test_tracking_resource_adaptor():
    cuda_mr = rmm.mr.CudaMemoryResource()

    mr = rmm.mr.TrackingResourceAdaptor(cuda_mr, capture_stacks=True)

    rmm.mr.set_current_device_resource(mr)

    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert mr.get_allocated_bytes() == 5040

    # Push a new Tracking adaptor
    mr2 = rmm.mr.TrackingResourceAdaptor(mr, capture_stacks=True)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.get_allocated_bytes() == 2016
    assert mr.get_allocated_bytes() == 7056

    # Ensure we get back a non-empty string for the allocations
    assert len(mr.get_outstanding_allocations_str()) > 0

    del buffers
    gc.collect()

    assert mr2.get_allocated_bytes() == 0
    assert mr.get_allocated_bytes() == 0

    # make sure the allocations string is now empty
    assert len(mr2.get_outstanding_allocations_str()) == 0
    assert len(mr.get_outstanding_allocations_str()) == 0


def test_failure_callback_resource_adaptor():
    retried = [False]

    def callback(nbytes: int) -> bool:
        if retried[0]:
            return False
        else:
            retried[0] = True
            return True

    cuda_mr = rmm.mr.CudaMemoryResource()
    mr = rmm.mr.FailureCallbackResourceAdaptor(cuda_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=int(1e11))
    assert retried[0]


def test_failure_callback_resource_adaptor_error():
    def callback(nbytes: int) -> bool:
        raise RuntimeError("MyError")

    cuda_mr = rmm.mr.CudaMemoryResource()
    mr = rmm.mr.FailureCallbackResourceAdaptor(cuda_mr, callback)
    rmm.mr.set_current_device_resource(mr)

    with pytest.raises(RuntimeError, match="MyError"):
        rmm.DeviceBuffer(size=int(1e11))


def test_dev_buf_circle_ref_dealloc():
    # This test creates a reference cycle containing a `DeviceBuffer`
    # and ensures that the garbage collector does not clear it, i.e.,
    # that the GC does not remove all references to other Python
    # objects from it. The `DeviceBuffer` needs to keep its reference
    # to the `DeviceMemoryResource` that was used to create it in
    # order to be cleaned up properly. See GH #931.

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    dbuf1 = rmm.DeviceBuffer(size=1_000_000)

    # Make dbuf1 part of a reference cycle:
    l1 = [dbuf1]
    l1.append(l1)

    # due to the reference cycle, the device buffer doesn't actually get
    # cleaned up until after `gc.collect()` is called.
    del dbuf1, l1

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    # test that after the call to `gc.collect()`, the `DeviceBuffer`
    # is deallocated successfully (i.e., without a segfault).
    gc.collect()


def test_upstream_mr_circle_ref_dealloc():
    # This test is just like the one above, except it tests that
    # instances of `UpstreamResourceAdaptor` (such as
    # `PoolMemoryResource`) are not cleared by the GC.

    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    mr = rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
    l1 = [mr]
    l1.append(l1)
    del mr, l1
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    gc.collect()


def test_mr_allocate_deallocate():
    mr = rmm.mr.TrackingResourceAdaptor(rmm.mr.get_current_device_resource())
    size = 1 << 23  # 8 MiB
    ptr = mr.allocate(size)
    assert mr.get_allocated_bytes() == 1 << 23
    mr.deallocate(ptr, size)
    assert mr.get_allocated_bytes() == 0


def test_custom_mr(capsys):
    base_mr = rmm.mr.CudaMemoryResource()

    def allocate_func(size, stream):
        print(f"Allocating {size} bytes")
        return base_mr.allocate(size, stream)

    def deallocate_func(ptr, size, stream):
        print(f"Deallocating {size} bytes")
        return base_mr.deallocate(ptr, size, stream)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    rmm.DeviceBuffer(size=256)

    captured = capsys.readouterr()
    assert captured.out == "Allocating 256 bytes\nDeallocating 256 bytes\n"


@pytest.mark.parametrize(
    "err_raise,err_catch",
    [
        (MemoryError, MemoryError),
        (RuntimeError, RuntimeError),
        (Exception, RuntimeError),
        (BaseException, RuntimeError),
    ],
)
def test_callback_mr_error(err_raise, err_catch):
    base_mr = rmm.mr.CudaMemoryResource()

    def allocate_func(size, stream):
        raise err_raise("My alloc error")

    def deallocate_func(ptr, size, stream):
        return base_mr.deallocate(ptr, size)

    rmm.mr.set_current_device_resource(
        rmm.mr.CallbackMemoryResource(allocate_func, deallocate_func)
    )

    with pytest.raises(err_catch, match="My alloc error"):
        rmm.DeviceBuffer(size=256)


@pytest.fixture
def make_reinit_hook():
    funcs = []

    def _make_reinit_hook(func, *args, **kwargs):
        funcs.append(func)
        rmm.register_reinitialize_hook(func, *args, **kwargs)
        return func

    yield _make_reinit_hook
    for func in funcs:
        rmm.unregister_reinitialize_hook(func)


def test_reinit_hooks_register(make_reinit_hook):
    L = []
    make_reinit_hook(lambda: L.append(1))
    make_reinit_hook(lambda: L.append(2))
    make_reinit_hook(lambda x: L.append(x), 3)

    rmm.reinitialize()
    assert L == [3, 2, 1]


def test_reinit_hooks_unregister(make_reinit_hook):
    L = []
    one = make_reinit_hook(lambda: L.append(1))
    make_reinit_hook(lambda: L.append(2))

    rmm.unregister_reinitialize_hook(one)
    rmm.reinitialize()
    assert L == [2]


def test_reinit_hooks_register_twice(make_reinit_hook):
    L = []

    def func_with_arg(x):
        L.append(x)

    def func_without_arg():
        L.append(2)

    make_reinit_hook(func_with_arg, 1)
    make_reinit_hook(func_without_arg)
    make_reinit_hook(func_with_arg, 3)
    make_reinit_hook(func_without_arg)

    rmm.reinitialize()
    assert L == [2, 3, 2, 1]


def test_reinit_hooks_unregister_twice_registered(make_reinit_hook):
    # unregistering a twice-registered function
    # should unregister both instances:
    L = []

    def func_with_arg(x):
        L.append(x)

    make_reinit_hook(func_with_arg, 1)
    make_reinit_hook(lambda: L.append(2))
    make_reinit_hook(func_with_arg, 3)

    rmm.unregister_reinitialize_hook(func_with_arg)
    rmm.reinitialize()
    assert L == [2]


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


@pytest.mark.parametrize("level", rmm.logging_level)
def test_valid_logging_level(level):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="RMM will not log logging_level.TRACE."
        )
        warnings.filterwarnings(
            "ignore", message="RMM will not log logging_level.DEBUG."
        )
        rmm.set_logging_level(level)
        assert rmm.get_logging_level() == level
        rmm.set_logging_level(rmm.logging_level.INFO)  # reset to default

        rmm.set_flush_level(level)
        assert rmm.get_flush_level() == level
        rmm.set_flush_level(rmm.logging_level.INFO)  # reset to default

        rmm.should_log(level)


@pytest.mark.parametrize(
    "level", ["INFO", 3, "invalid", 100, None, 1.2345, [1, 2, 3]]
)
def test_invalid_logging_level(level):
    with pytest.raises(TypeError):
        rmm.set_logging_level(level)
    with pytest.raises(TypeError):
        rmm.set_flush_level(level)
    with pytest.raises(TypeError):
        rmm.should_log(level)
