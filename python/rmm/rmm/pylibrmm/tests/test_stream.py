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


def test_cuda_stream_protocol():
    device = Device()
    device.set_current()
    rmm_stream = rmm.pylibrmm.stream.Stream()

    # RMM -> cuda.core
    cuda_stream = device.create_stream(rmm_stream)
    assert cuda_stream is not None

    # cuda.core -> RMM
    rmm_stream_2 = rmm.pylibrmm.stream.Stream(cuda_stream)
    assert rmm_stream_2.__cuda_stream__() == cuda_stream.__cuda_stream__()


def test_cuda_core_stream_to_rmm():
    device = Device()
    device.set_current()
    cuda_stream = device.create_stream()

    rmm_stream = rmm.pylibrmm.stream.Stream(cuda_stream)
    cuda_stream_2 = device.create_stream(rmm_stream)

    assert cuda_stream_2.__cuda_stream__() == cuda_stream_2.__cuda_stream__()


def test_cuda_stream_from_cuda_core_default_stream():
    device = Device()
    device.set_current()

    rmm_stream = rmm.pylibrmm.stream.Stream(device.default_stream)
    assert (
        rmm_stream.__cuda_stream__() == device.default_stream.__cuda_stream__()
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
