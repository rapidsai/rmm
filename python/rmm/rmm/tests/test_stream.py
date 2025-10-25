# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata

import packaging.version
import pytest
from cuda.core.experimental import Device

import rmm.pylibrmm.stream

CUDA_CORE_VERSION = importlib.metadata.version("cuda-core")
CUDA_CORE_0_4_0 = packaging.version.parse(
    CUDA_CORE_VERSION
) >= packaging.version.parse("0.4.0")


@pytest.fixture
def current_device():
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


@pytest.mark.skipif(not CUDA_CORE_0_4_0, reason="cuda.core >=0.4.0 required.")
def test_cuda_core_buffer(current_device):
    # Test that RMM's Stream duck-types as a cuda.core.Stream
    from cuda.core.experimental import DeviceMemoryResource

    rmm_stream = rmm.pylibrmm.stream.Stream()
    cuda_core_mr = DeviceMemoryResource(device_id=current_device.device_id)

    buf = cuda_core_mr.allocate(1024, stream=rmm_stream)
    buf.close(stream=rmm_stream)
    rmm_stream.synchronize()
