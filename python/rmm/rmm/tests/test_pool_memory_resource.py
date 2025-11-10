# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PoolMemoryResource."""

import pytest
from numba import cuda
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm
from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_pool_memory_resource(dtype, nelem, alloc):
    mr = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size="4MiB",
        maximum_pool_size="8MiB",
    )
    rmm.mr.set_current_device_resource(mr)
    assert rmm.mr.get_current_device_resource_type() is type(mr)
    array_tester(dtype, nelem, alloc)


def test_reinitialize_max_pool_size():
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size="8MiB"
    )
    rmm.DeviceBuffer().resize((1 << 23) - 1)


def test_reinitialize_max_pool_size_exceeded():
    rmm.reinitialize(
        pool_allocator=True, initial_pool_size=0, maximum_pool_size=1 << 23
    )
    with pytest.raises(MemoryError):
        rmm.DeviceBuffer().resize(1 << 24)


@pytest.mark.parametrize("stream", [cuda.default_stream(), cuda.stream()])
def test_rmm_pool_numba_stream(stream):
    rmm.reinitialize(pool_allocator=True)

    stream = Stream(stream)
    a = rmm.DeviceBuffer(size=3, stream=stream)

    assert a.size == 3
    assert a.ptr != 0


def test_mr_upstream_lifetime():
    # Simple test to ensure upstream MRs are deallocated before downstream MR
    cuda_mr = rmm.mr.CudaMemoryResource()

    pool_mr = rmm.mr.PoolMemoryResource(cuda_mr)

    # Delete cuda_mr first. Should be kept alive by pool_mr
    del cuda_mr
    del pool_mr
