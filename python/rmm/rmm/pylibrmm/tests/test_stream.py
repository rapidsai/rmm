# Copyright (c) 2025, NVIDIA CORPORATION.
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


import cupy as cp
import pytest
from cuda.core.experimental import Device

import rmm.pylibrmm.stream


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


def test_cuda_stream_cupy():
    cupy_stream = cp.cuda.Stream()
    rmm_stream = rmm.pylibrmm.stream.Stream(cupy_stream)

    assert rmm_stream.__cuda_stream__() == (0, cupy_stream.ptr)


def test_cuda_core_vector_add(current_device):
    # https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/vector_add.py
    # but with a stream wrapping an RMM stream

    from cuda.core.experimental import (
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )

    rmm_stream = rmm.pylibrmm.stream.Stream()

    # compute c = a + b
    code = """
    template<typename T>
    __global__ void vector_add(const T* A,
                            const T* B,
                            T* C,
                            size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
            C[i] = A[i] + B[i];
        }
    }
    """

    current_device

    # prepare program
    program_options = ProgramOptions(
        std="c++17", arch=f"sm_{current_device.arch}"
    )
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("vector_add<float>",))

    # run in single precision
    ker = mod.get_kernel("vector_add<float>")
    dtype = cp.float32

    # prepare input/output
    size = 5000
    rng = cp.random.default_rng()
    a = rng.random(size, dtype=dtype)
    b = rng.random(size, dtype=dtype)
    c = cp.empty_like(a)

    # cupy runs on a different stream from s, so sync before accessing
    current_device.sync()

    # prepare launch
    block = 256
    grid = (size + block - 1) // block
    config = LaunchConfig(grid=grid, block=block)

    # launch kernel on stream s
    launch(
        rmm_stream,
        config,
        ker,
        a.data.ptr,
        b.data.ptr,
        c.data.ptr,
        cp.uint64(size),
    )
    cuda_stream = current_device.create_stream(rmm_stream)
    cuda_stream.sync()

    # check result
    assert cp.allclose(c, a + b)
