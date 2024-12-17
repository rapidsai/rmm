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

import numba.cuda as numba_cuda
import numpy as np
import pytest

import rmm
import rmm._cuda.stream
from rmm.allocators.numba import RMMNumbaManager


@pytest.fixture(scope="session")
def numba_allocator():
    numba_cuda.set_memory_manager(RMMNumbaManager)


def test_rmm_device_buffer_copy_from_numba_device_array():
    cuda_ary = numba_cuda.to_device(np.array([97, 98, 99], dtype="u1"))
    db = rmm.DeviceBuffer.to_device(np.zeros(10, dtype="u1"))
    db.copy_from_device(cuda_ary)

    expected = np.array([97, 98, 99, 0, 0, 0, 0, 0, 0, 0], dtype="u1")
    result = db.copy_to_host()

    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    "stream",
    [lambda: numba_cuda.default_stream(), lambda: numba_cuda.stream()],
)
def test_rmm_pool_numba_stream(stream):
    rmm.reinitialize(pool_allocator=True)
    stream = rmm._cuda.stream.Stream(stream())
    a = rmm.pylibrmm.device_buffer.DeviceBuffer(size=3, stream=stream)

    assert a.size == 3
    assert a.ptr != 0


@pytest.mark.parametrize(
    "make_copy", [lambda db: db.copy(), lambda db: copy.copy(db)]
)
def test_numba_rmm_device_buffer_copy(make_copy):
    cuda_ary = numba_cuda.to_device(np.array([97, 98, 99, 0, 0], dtype="u1"))
    db = rmm.DeviceBuffer.to_device(np.zeros(5, dtype="u1"))
    db.copy_from_device(cuda_ary)
    db_copy = make_copy(db)

    assert db is not db_copy
    assert db.ptr != db_copy.ptr
    assert len(db) == len(db_copy)

    expected = np.array([97, 98, 99, 0, 0], dtype="u1")
    result = db_copy.copy_to_host()

    np.testing.assert_equal(expected, result)
