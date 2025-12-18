# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pytest

import rmm.pylibrmm.cuda_stream
import rmm.pylibrmm.cuda_stream_pool
import rmm.pylibrmm.stream


@pytest.fixture
def current_device():
    pytest.importorskip("cuda.core", minversion="0.5.0")
    from cuda.core import Device

    device = Device()
    device.set_current()
    return device


def test_cuda_stream_protocol(current_device):
    rmm_stream = rmm.pylibrmm.stream.Stream()

    # RMM -> cuda.core
    cuda_stream = current_device.create_stream(rmm_stream)
    assert cuda_stream is not None

    # cuda.core -> RMM
    rmm_stream_2 = rmm.pylibrmm.stream.Stream(cuda_stream)
    assert rmm_stream_2.__cuda_stream__() == cuda_stream.__cuda_stream__()


def test_cuda_core_stream_to_rmm(current_device):
    cuda_stream = current_device.create_stream()

    rmm_stream = rmm.pylibrmm.stream.Stream(cuda_stream)
    cuda_stream_2 = current_device.create_stream(rmm_stream)

    assert cuda_stream_2.__cuda_stream__() == cuda_stream_2.__cuda_stream__()


def test_rmm_stream_from_cuda_core_default_stream(current_device):
    rmm_stream = rmm.pylibrmm.stream.Stream(current_device.default_stream)
    assert (
        rmm_stream.__cuda_stream__()
        == current_device.default_stream.__cuda_stream__()
    )


def test_cuda_core_stream_from_rmm_default_stream(current_device):
    cuda_stream = current_device.create_stream(
        rmm.pylibrmm.stream.DEFAULT_STREAM
    )
    assert (
        cuda_stream.__cuda_stream__()
        == rmm.pylibrmm.stream.DEFAULT_STREAM.__cuda_stream__()
    )


def test_cuda_stream_protocol_not_supported():
    class V1Stream:
        def __cuda_stream__(self):
            return (1, 2)

    obj = V1Stream()
    with pytest.raises(NotImplementedError, match="version: '1'"):
        rmm.pylibrmm.stream.Stream(obj)


def test_cuda_stream_cupy(current_device):
    cp = pytest.importorskip("cupy")
    cupy_stream = cp.cuda.Stream()
    rmm_stream = rmm.pylibrmm.stream.Stream(cupy_stream)

    assert rmm_stream.__cuda_stream__() == (0, cupy_stream.ptr)
    cuda_stream = current_device.create_stream(rmm_stream)
    assert cuda_stream.__cuda_stream__() == (0, cupy_stream.ptr)


def test_cuda_core_buffer(current_device):
    # Test that RMM's Stream duck-types as a cuda.core.Stream
    pytest.importorskip("cuda.core", minversion="0.5.0")
    from cuda.core import DeviceMemoryResource

    rmm_stream = current_device.create_stream(rmm.pylibrmm.stream.Stream())
    cuda_core_mr = DeviceMemoryResource(device_id=current_device.device_id)

    buf = cuda_core_mr.allocate(1024, stream=rmm_stream)
    buf.close(stream=rmm_stream)
    rmm_stream.synchronize()


@pytest.mark.parametrize(
    "flags",
    [
        rmm.pylibrmm.cuda_stream.CudaStreamFlags.SYNC_DEFAULT,
        rmm.pylibrmm.cuda_stream.CudaStreamFlags.NON_BLOCKING,
    ],
)
def test_cuda_stream_pool(current_device, flags):
    default_rmm_stream = rmm.pylibrmm.stream.Stream(
        current_device.default_stream
    )

    stream_pool = rmm.pylibrmm.cuda_stream_pool.CudaStreamPool(
        pool_size=10, flags=flags
    )
    assert stream_pool.get_pool_size() == 10

    streams = [stream_pool.get_stream() for _ in range(10)]

    for i in range(10):
        for j in range(i + 1, 10):
            assert streams[i] != streams[j]
        # should not be the default stream
        assert streams[i] != default_rmm_stream
        assert streams[i] == stream_pool.get_stream(i)


def test_hashable():
    a = rmm.pylibrmm.stream.Stream()
    b = rmm.pylibrmm.stream.Stream()
    assert hash(a) == hash(a)
    assert hash(a) != hash(b)

    assert a == a
    assert a != b

    assert len({a, b}) == 2

    a2 = rmm.pylibrmm.stream.Stream(a)
    assert a2 == a
    assert hash(a2) == hash(a)
