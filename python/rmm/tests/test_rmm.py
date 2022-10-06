# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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


@pytest.mark.skipif(
    not _CUDAMALLOC_ASYNC_SUPPORTED,
    reason="cudaMallocAsync not supported",
)
def test_cuda_async_memory_resource_ipc():
    # Test that enabling IPC earlier than CUDA 11.3 raises a ValueError
    if _driver_version < 11030 or _runtime_version < 11030:
        with pytest.raises(ValueError):
            mr = rmm.mr.CudaAsyncMemoryResource(enable_ipc=True)
    else:
        try:
            mr = rmm.mr.CudaAsyncMemoryResource(enable_ipc=True)
        except RuntimeError:
            raise RuntimeError(
                f"drv: {_driver_version}, rt: {_runtime_version}"
            )
        rmm.mr.set_current_device_resource(mr)
        assert rmm.mr.get_current_device_resource_type() is type(mr)

